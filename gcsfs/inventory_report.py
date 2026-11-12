import asyncio
import json
import re
import zlib
from dataclasses import dataclass
from datetime import date, datetime

from .retry import validate_response

# Default streaming limits. They are intentionally conservative so that a
# corrupt or adversarial shard cannot make the reader grow without bounds.
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB network/decompression chunks
DEFAULT_MAX_DECOMPRESSED_BYTES = 5 * 2**30  # 5 GiB per gzip shard
DEFAULT_MAX_RECORD_BYTES = 64 * 2**20  # 64 MiB for a single CSV record
DEFAULT_RECORD_QUEUE_SIZE = 1000  # buffered parsed records per snapshot
DEFAULT_HEAD_CONCURRENCY = 4

_MANIFEST_SUFFIX = "_manifest.json"
_MANIFEST_NAME_RE = re.compile(
    r"^(?P<config_id>.+)_(?P<target_date>\d{4}-\d{2}-\d{2})"
    r"T(?P<target_time>\d{2}:\d{2})_manifest\.json$"
)


class InventoryReportError(ValueError):
    """Base class for inventory report fetching and parsing failures."""


class InventoryReportIntegrityError(InventoryReportError):
    """Raised when a report batch cannot be proven complete or consistent.

    This covers missing shards, shards from different report times, manifest
    inconsistencies and schema drift between the report config and the
    manifest. The reader fails explicitly instead of pretending a partial set
    of shards is a complete snapshot.
    """


class InventoryReportParseError(InventoryReportError):
    """Raised when the content of a shard cannot be parsed.

    The error always carries the shard object name and, when available, the
    zero-based data record index and the byte offset within the (decompressed)
    shard where the failure was detected.
    """

    def __init__(
        self,
        message,
        *,
        shard=None,
        record_index=None,
        byte_offset=None,
    ):
        self.shard = shard
        self.record_index = record_index
        self.byte_offset = byte_offset
        location = []
        if shard is not None:
            location.append(f"shard={shard}")
        if record_index is not None:
            location.append(f"record_index={record_index}")
        if byte_offset is not None:
            location.append(f"byte_offset={byte_offset}")
        suffix = f" ({', '.join(location)})" if location else ""
        super().__init__(f"{message}{suffix}")


class InventoryReportLimitError(InventoryReportError):
    """Raised when a decompression or single-record size limit is exceeded."""

    def __init__(
        self,
        message,
        *,
        shard=None,
        record_index=None,
        byte_offset=None,
        limit=None,
        actual=None,
    ):
        self.shard = shard
        self.record_index = record_index
        self.byte_offset = byte_offset
        self.limit = limit
        self.actual = actual
        details = [message]
        if limit is not None:
            details.append(f"limit={limit}")
        if actual is not None:
            details.append(f"actual={actual}")
        location = []
        if shard is not None:
            location.append(f"shard={shard}")
        if record_index is not None:
            location.append(f"record_index={record_index}")
        if byte_offset is not None:
            location.append(f"byte_offset={byte_offset}")
        suffix = f" ({', '.join(details + location)})"
        super().__init__(suffix)


@dataclass(frozen=True)
class PinnedShard:
    """A single shard whose content version has been pinned via generation.

    Attributes:
        name: full object name of the shard in the destination bucket.
        generation: pinned object generation; media reads must use it so the
            content cannot change underneath the reader after pinning.
        size: shard object size in bytes as listed, if available.
        compressed: whether the shard payload is gzip compressed.
        shard_index: position of the shard in the manifest.
    """

    name: str
    generation: str
    size: int | None
    compressed: bool
    shard_index: int


@dataclass(frozen=True)
class PinnedReportBatch:
    """One explicitly selected, integrity-validated report batch."""

    source_bucket: str
    destination_bucket: str
    manifest_name: str
    snapshot_time: str
    target_datetime: str
    records_processed: int | None
    shards: tuple[PinnedShard, ...]
    csv_options: dict
    metadata_fields: tuple[str, ...]


class InventoryReport:
    """
    A utility class for fetching and processing inventory reports from GCS.

    The 'InventoryReport' class provides logic to support logic to fetch
    inventory reports, and process their content to obtain a final snapshot
    of objects in the latest inventory reports.

    High-Level Functionality:
    ------------------------
    1. Fetching Inventory Reports:
       - The class offers methods to fetch inventory report configurations and
         metadata from GCS.
       - It validates the inventory report information provided by the user.
       - Inventory report configurations include options for parsing CSV format
         and specifying the bucket and destination path.

    2. Parsing and Processing Inventory Report Content:
       - The class processes the raw content of inventory reports to extract
         object details such as name, size, etc.
       - It supports listing objects using a snapshot option or filtering
         based on a user-defined prefix.
       - The class handles CSV parsing, removes header (if specified), and
         fetches required object metadata.

    3. Constructing the Final Snapshot:
       - If the user wishes to use the snapshot to do listing directly, the
         snapshot will contain the relevant object details and subdirectory
         prefixes, filtered by the prefix.

       - If the user wishes to use the snapshot as a starting point for async
         listing, the snapshot will only contain a list of object names,
         filtered by the prefix.

    Note:
    -----
    - The class should only be internally used in the 'GCSFileSystem' as an
      optional configuration during listing.

    Example Usage:
    --------------
    # Should already be instanted in 'core.py'
    gcs_file_system = GCSFileSystem(...)

    # User defines inventory report information
    inventory_report_info = {
        "use_snapshot_listing": True,
        "location": "us-east1",
        "id": "inventory_report_id"
    }

    # User defines a prefix for filtering objects
    prefix = "prefix/"

    # Fetch the snapshot based on inventory reports
    items, prefixes = await InventoryReport.fetch_snapshot(
    gcs_file_system, inventory_report_info, prefix)
    """

    # HTTP endpoint of the Storage Insights Service.
    BASE_URL = "https://storageinsights.googleapis.com/v1"

    @classmethod
    async def fetch_snapshot(cls, gcs_file_system, inventory_report_info, prefix):
        """
        Main entry point of the 'InventoryReport' class.
        Fetches the latest snapshot of objects based on inventory report configuration.

        Parameters:
            gcs_file_system (GCSFileSystem): An instance of the 'GCSFileSystem'
            class (see 'core.py').
            inventory_report_info (dict): A client-configured dictionary
            containing inventory report information.
            prefix (str): Listing prefix specified by the client.

        Returns:
            tuple: A tuple containing two lists: the 'items' list representing
            object details for the snapshot, and the 'prefixes' list containing
            subdirectory prefixes.

            Note: when 'use_snapshot_listing' in 'inventory_report_info' is set
            to False, the 'prefixes' list will be empty, and the 'items' list
            will contain only the object names.
        """
        # Validate the inventory report info that the user passes in.
        cls._validate_inventory_report_info(inventory_report_info)

        # Parse the inventory report info.
        use_snapshot_listing = inventory_report_info.get("use_snapshot_listing")
        inventory_report_location = inventory_report_info.get("location")
        inventory_report_id = inventory_report_info.get("id")

        # Fetch the inventory report configuration.
        raw_inventory_report_config = await cls._fetch_raw_inventory_report_config(
            gcs_file_system=gcs_file_system,
            location=inventory_report_location,
            id=inventory_report_id,
        )

        # Parse the inventory report configuration.
        inventory_report_config = cls._parse_raw_inventory_report_config(
            raw_inventory_report_config=raw_inventory_report_config,
            use_snapshot_listing=use_snapshot_listing,
        )

        # Use the config to fetch all inventory report metadata.
        unsorted_inventory_report_metadata = await cls._fetch_inventory_report_metadata(
            gcs_file_system=gcs_file_system,
            inventory_report_config=inventory_report_config,
        )

        # Sort the metadata based on reverse created time order.
        inventory_report_metadata = cls._sort_inventory_report_metadata(
            unsorted_inventory_report_metadata=unsorted_inventory_report_metadata
        )

        # Download the most recent inventory reports in raw form.
        bucket = inventory_report_config.bucket
        inventory_report_content = await cls._download_inventory_report_content(
            gcs_file_system=gcs_file_system,
            inventory_report_metadata=inventory_report_metadata,
            bucket=bucket,
        )

        # Parse the raw inventory reports into snapshot objects.
        objects = cls._parse_inventory_report_content(
            gcs_file_system=gcs_file_system,
            inventory_report_content=inventory_report_content,
            inventory_report_config=inventory_report_config,
            use_snapshot_listing=use_snapshot_listing,
            bucket=bucket,
        )

        # Construct the final snapshot based on the fetched objects.
        snapshot = cls._construct_final_snapshot(
            objects=objects, prefix=prefix, use_snapshot_listing=use_snapshot_listing
        )

        # Return the final snapshot.
        return snapshot

    def _validate_inventory_report_info(inventory_report_info):
        """
        Validates the inventory report information dictionary that user
        passes in.

        Parameters:
            inventory_report_info (dict): A dictionary containing the inventory
            report information with the following keys:
                - "use_snapshot_listing" (bool): A flag indicating whether
                  to use snapshot listing in the inventory report.
                - "location" (str): The location of the inventory report in GCS.
                - "id" (str): The ID of the inventory report in GCS.

        Raises:
            ValueError: If any required key (use_snapshot_listing, location, id)
            is missing from the inventory_report_info dictionary.
        """
        if "use_snapshot_listing" not in inventory_report_info:
            raise ValueError("Use snapshot listing is not configured.")
        if "location" not in inventory_report_info:
            raise ValueError("Inventory report location is not configured.")
        if "id" not in inventory_report_info:
            raise ValueError("Inventory report id is not configured.")

    async def _fetch_raw_inventory_report_config(gcs_file_system, location, id):
        """
        Fetches the raw inventory report configuration from GCS based on the
        specified location and ID.

        Parameters:
            gcs_file_system (GCSFileSystem): An instance of the 'GCSFileSystem'
            class (see 'core.py').
            location (str): The location of the inventory report in GCS.
            id (str): The ID of the inventory report in GCS.

        Returns:
            dict: A dictionary containing the raw inventory report
            configuration retrieved from GCS.

        Raises:
            Exception: If there is an error while fetching the inventory
            report configuration.
        """
        project = gcs_file_system.project
        url = "{}/projects/{}/locations/{}/reportConfigs/{}"
        url = url.format(InventoryReport.BASE_URL, project, location, id)
        try:
            raw_inventory_report_config = await gcs_file_system._call(
                "GET", url, json_out=True
            )
            return raw_inventory_report_config
        except Exception as e:
            raise ValueError(
                f"Error encountered when fetching inventory report config: {e}."
            )

    def _parse_raw_inventory_report_config(
        raw_inventory_report_config,
        use_snapshot_listing,
        require_size=True,
    ):
        """
        Parses the raw inventory report configuration and validates its properties.

        Parameters:
            raw_inventory_report_config (dict): A dictionary containing the raw
            inventory report configuration retrieved from GCS.
            use_snapshot_listing (bool): A flag indicating whether to use snapshot
            listing in the inventory report.

        Returns:
            InventoryReportConfig: An instance of the InventoryReportConfig
            class representing the parsed inventory report configuration.

        Raises:
            ValueError: If the current date is outside the start and
            end range specified in the inventory report config.
            ValueError: If the "name" field is not present in the metadata
            fields of the report config.
            ValueError: If "size" field is not present in the metadata
            fields and use_snapshot_listing is True.
        """
        # Parse the report config.
        frequency_options = raw_inventory_report_config.get("frequencyOptions")
        start_date = InventoryReport._convert_obj_to_date(
            frequency_options.get("startDate")
        )
        end_date = InventoryReport._convert_obj_to_date(
            frequency_options.get("endDate")
        )
        object_metadata_report_options = raw_inventory_report_config.get(
            "objectMetadataReportOptions"
        )
        storage_destination_options = object_metadata_report_options.get(
            "storageDestinationOptions"
        )

        # Save relevant report config properties.
        csv_options = raw_inventory_report_config.get("csvOptions")
        bucket = storage_destination_options.get("bucket")
        destination_path = storage_destination_options.get("destinationPath")
        metadata_fields = object_metadata_report_options.get("metadataFields")

        # Validate date, making sure the current date is within the start and end range.
        today = datetime.now()
        if today < start_date or today > end_date:
            raise ValueError(
                f"Current date {today} is outside the range \
                {start_date} and {end_date} specified by the inventory report config."
            )

        # Validate object name exists in the metadata fields.
        # Note that the size field is mandated to be included in the
        # config when the client sets up the inventory report.
        obj_name_idx = metadata_fields.index("name")

        # If the user wants to do listing based on the snapshot, also
        # validate the report contains size metadata for each object -- unless
        # the caller explicitly opted into fetching missing metadata with
        # per-object HEAD requests.
        if use_snapshot_listing and require_size:
            try:
                metadata_fields.index("size")
            except ValueError:
                raise ValueError(
                    "If you want to use the snapshot for listing, the object size \
                        metadata has to be included in the inventory report."
                )

        # Finally, construct and return the inventory report config.
        inventory_report_config = InventoryReportConfig(
            csv_options=csv_options,
            bucket=bucket,
            destination_path=destination_path,
            metadata_fields=metadata_fields,
            obj_name_idx=obj_name_idx,
        )

        return inventory_report_config

    async def _fetch_inventory_report_metadata(
        gcs_file_system, inventory_report_config
    ):
        """
        Fetches all inventory report metadata from GCS based on the specified
        inventory report config.

        Parameters:
            gcs_file_system (GCSFileSystem): An instance of the 'GCSFileSystem'
            class (see 'core.py').
            inventory_report_config (InventoryReportConfig): An instance of
            the InventoryReportConfig class representing the inventory report
            configuration.

        Returns:
            list: A list containing dictionaries representing the metadata of
            objects from the inventory reports.

        Raises:
            ValueError: If the fetched inventory reports are empty.
        """
        # There might be multiple inventory reports in the bucket.
        inventory_report_metadata = []

        # Extract out bucket and destination path of the inventory reports.
        bucket = inventory_report_config.bucket
        destination_path = inventory_report_config.destination_path

        # Fetch the first page.
        page = await gcs_file_system._call(
            "GET", "b/{}/o", bucket, prefix=destination_path, json_out=True
        )

        inventory_report_metadata.extend(page.get("items", []))
        next_page_token = page.get("nextPageToken", None)

        # Keep fetching new pages as long as next page token exists.
        # Note that the iteration in the while loop should most likely
        # be minimal. For reference, a million objects is split up into
        # two reports, and if the report is generated daily, then in a year,
        # there will be roughly ~700 reports generated, which will still be
        # fetched in a single page.
        while next_page_token is not None:
            page = await gcs_file_system._call(
                "GET",
                "b/{}/o",
                bucket,
                prefix=destination_path,
                json_out=True,
                pageToken=next_page_token,
            )

            inventory_report_metadata.extend(page.get("items", []))
            next_page_token = page.get("nextPageToken", None)

        # If no reports are fetched, indicates there is an error.
        if len(inventory_report_metadata) == 0:
            raise ValueError(
                "No inventory reports to fetch. Check if \
                your inventory report is set up correctly."
            )

        return inventory_report_metadata

    def _sort_inventory_report_metadata(unsorted_inventory_report_metadata):
        """
        Sorts the inventory report metadata based on the 'timeCreated' field
        in reverse chronological order.

        Parameters:
            unsorted_inventory_report_metadata (list): A list of dictionaries
            representing the metadata of objects from the inventory reports.

        Returns:
            list: A sorted list of dictionaries representing the inventory
            report metadata, sorted in reverse chronological order based
            on 'timeCreated'.
        """
        return sorted(
            unsorted_inventory_report_metadata,
            key=lambda ir: InventoryReport._convert_str_to_datetime(
                ir.get("timeCreated")
            ),
            reverse=True,
        )

    async def _download_inventory_report_content(
        gcs_file_system, inventory_report_metadata, bucket
    ):
        """
        Downloads the most recent inventory report content from GCS based on
        the inventory report metadata.

        Parameters:
            gcs_file_system (GCSFileSystem): An instance of the 'GCSFileSystem'
            class (see 'core.py').
            inventory_report_metadata (list): A list of dictionaries
            representing the metadata of objects from the inventory reports.
            bucket (str): The name of the GCS bucket containing
            the inventory reports.

        Returns:
            list: A list containing the content of the most recent inventory
            report as strings.
        """
        # Get the most recent inventory report date.
        most_recent_inventory_report = inventory_report_metadata[0]
        most_recent_date = InventoryReport._convert_str_to_datetime(
            most_recent_inventory_report.get("timeCreated")
        ).date()

        inventory_report_content = []

        # Run a for loop here, since there might be multiple inventory reports
        # generated on the same day. For reference, 1 million objects will be
        # split into only 2 inventory reports, so it is very rare that there
        # will be many inventory reports on the same day. But including this
        # logic for robustness.
        for metadata in inventory_report_metadata:
            inventory_report_date = InventoryReport._convert_str_to_datetime(
                metadata["timeCreated"]
            ).date()

            if inventory_report_date == most_recent_date:
                # Download the raw inventory report if the date matches.
                # Header is not needed, we only need to process and store
                # the content.
                _header, encoded_content = await gcs_file_system._call(
                    "GET", "b/{}/o/{}", bucket, metadata.get("name"), alt="media"
                )

                # Decode the binary content into string for the content.
                decoded_content = encoded_content.decode()

                inventory_report_content.append(decoded_content)

        return inventory_report_content

    def _parse_inventory_report_content(
        gcs_file_system,
        inventory_report_content,
        inventory_report_config,
        use_snapshot_listing,
        bucket,
    ):
        """
        Parses the raw inventory report content and extracts object details.

        Parameters:
            gcs_file_system (GCSFileSystem): An instance of the 'GCSFileSystem'
            class (see 'core.py').
            inventory_report_content (list): A list of strings containing the
            raw content of the inventory report.
            inventory_report_config (InventoryReportConfig): An instance of the
            InventoryReportConfig class representing the inventory report
            configuration.
            use_snapshot_listing (bool): A flag indicating whether to use snapshot
            listing in the inventory report.
            bucket (str): The name of the GCS bucket containing the inventory
            reports.

        Returns:
            list: A list of dictionaries representing object details parsed
            from the inventory report content.
        """
        # Get the csv configuration for each inventory report.
        csv_options = inventory_report_config.csv_options
        record_separator = csv_options.get("recordSeparator", "\n")
        delimiter = csv_options.get("delimiter", ",")
        header_required = csv_options.get("headerRequired", False)

        objects = []

        for content in inventory_report_content:
            # Split the content into lines based on the specified separator.
            lines = content.split(record_separator)

            # Remove the header, if present.
            if header_required:
                lines = lines[1:]

            # Parse each line of the inventory report.
            for line in lines:
                obj = InventoryReport._parse_inventory_report_line(
                    inventory_report_line=line,
                    use_snapshot_listing=use_snapshot_listing,
                    gcs_file_system=gcs_file_system,
                    inventory_report_config=inventory_report_config,
                    delimiter=delimiter,
                    bucket=bucket,
                )

                objects.append(obj)

        return objects

    def _parse_inventory_report_line(
        inventory_report_line,
        use_snapshot_listing,
        gcs_file_system,
        inventory_report_config,
        delimiter,
        bucket,
    ):
        """
        Parses a single line of the inventory report and extracts object details.

        Parameters:
            inventory_report_line (str): A string representing a single line of
            the raw content from the inventory report.
            use_snapshot_listing (bool): A flag indicating whether to use snapshot
            listing in the inventory report.
            gcs_file_system (GCSFileSystem): An instance of the 'GCSFileSystem'
            class (see 'core.py').
            inventory_report_config (InventoryReportConfig): An instance of the
            InventoryReportConfig class representing the inventory report
            configuration.
            delimiter (str): The delimiter used in the inventory report content
            to separate fields.
            bucket (str): The name of the GCS bucket containing the inventory
            reports.

        Returns:
            dict: A dictionary representing object details parsed from the
            inventory report line.
        """
        obj_name_idx = inventory_report_config.obj_name_idx
        metadata_fields = inventory_report_config.metadata_fields

        # If the client wants to do listing from the snapshot, we need
        # to fetch all the metadata for each object. Otherwise, we only
        # need to fetch the name.
        if use_snapshot_listing is True:
            obj = gcs_file_system._process_object(
                bucket,
                {
                    key: value
                    for key, value in zip(
                        metadata_fields, inventory_report_line.strip().split(delimiter)
                    )
                },
            )
        else:
            obj = {"name": inventory_report_line.strip().split(delimiter)[obj_name_idx]}

        return obj

    def _construct_final_snapshot(objects, prefix, use_snapshot_listing):
        """
        Constructs the final snapshot based on the retrieved objects and prefix.

        Parameters:
            objects (list): A list of dictionaries representing object details
            from the inventory report.
            prefix (str): A prefix used to filter objects in the snapshot based
            on their names.
            use_snapshot_listing (bool): A flag indicating whether to use snapshot
            listing in the inventory report.

        Returns:
            tuple: A tuple containing two lists: the 'items' list representing
            object details for the snapshot, and the 'prefixes' list containing
            subdirectory prefixes. If 'use_snapshot_listing' is set to False,
            'prefix' will also be empty, and 'items' will contains the object
            names in the snapshot.
        """
        if prefix is None:
            prefix = ""

        # Filter the prefix and returns the list if the user does not want to use
        # the snapshot for listing.
        if use_snapshot_listing is False:
            return [obj for obj in objects if obj.get("name").startswith(prefix)], []

        else:
            # If the user wants to use the snapshot, generate both the items and
            # prefixes manually.
            items = []
            prefixes = set()

            for obj in objects:
                # Fetch the name of the object.
                obj_name = obj.get("name")

                # If the object name doesn't start with the prefix, continue.
                # In the case where prefix is empty, it will always return
                # true (which is the expected behavior).
                if not obj_name.startswith(prefix):
                    continue

                # Remove the prefix.
                object_name_no_prefix = obj_name[len(prefix) :]

                # Determine whether the object name is a directory.
                first_delimiter_idx = object_name_no_prefix.find("/")

                # If not, then append it to items.
                if first_delimiter_idx == -1:
                    items.append(obj)
                    continue

                # If it is, recompose the directory and add to the prefix set.
                dir = object_name_no_prefix[:first_delimiter_idx]
                obj_prefix = (
                    prefix.rstrip("/")
                    + ("" if prefix == "" else "/")
                    + dir
                    + ("" if dir == "" else "/")
                )
                prefixes.add(obj_prefix)

        return items, list(prefixes)

    @staticmethod
    def _convert_obj_to_date(obj):
        """
        Converts a dictionary representing a date object to a datetime object.

        Parameters:
            obj (dict): A dictionary representing a date object with keys "day",
            "month", and "year".

        Returns:
            datetime: A datetime object representing the converted date.
        """
        day = obj["day"]
        month = obj["month"]
        year = obj["year"]
        return datetime(year, month, day)

    @staticmethod
    def _convert_str_to_datetime(str):
        """
        Converts an ISO-formatted date string to a datetime object.

        Parameters:
            date_string (str): An ISO-formatted date string with or without
            timezone information (Z).

        Returns:
            datetime: A datetime object representing the converted date and time.
        """
        return datetime.fromisoformat(str.replace("Z", "+00:00"))

    # ------------------------------------------------------------------
    # Streaming snapshot entry point
    # ------------------------------------------------------------------

    @classmethod
    async def iter_snapshot(
        cls,
        gcs_file_system,
        inventory_report_info,
        prefix=None,
        *,
        report_datetime=None,
        chunk_size=DEFAULT_CHUNK_SIZE,
        max_decompressed_bytes=DEFAULT_MAX_DECOMPRESSED_BYTES,
        max_record_bytes=DEFAULT_MAX_RECORD_BYTES,
        head_missing_metadata=False,
        head_concurrency=DEFAULT_HEAD_CONCURRENCY,
        record_queue_size=DEFAULT_RECORD_QUEUE_SIZE,
        publish_dircache=False,
    ):
        """Stream object records from one explicitly selected report batch.

        Unlike :meth:`fetch_snapshot`, this never downloads every shard as a
        string and splits it into a full in-memory list. Shards are read in
        bounded chunks, gzip payloads are decompressed incrementally and CSV
        records are parsed by a stateful tokenizer, so memory usage stays
        bounded regardless of report size.

        The report batch (manifest plus every shard generation) is pinned and
        validated before any record is yielded. A missing shard, a shard from
        a different report time, or schema drift fails the iteration
        explicitly -- a partial shard set is never presented as a complete
        snapshot. The yielded records describe the bucket as it was at the
        manifest's ``snapshot_time``; they do not promise the current bucket
        state.

        Parameters:
            gcs_file_system (GCSFileSystem): filesystem used for GCS calls.
            inventory_report_info (dict): client configuration with
                ``location``, ``id`` and ``use_snapshot_listing``.
            prefix (str | None): object-key prefix filter, matched on key
                boundaries (``"dir"`` matches ``"dir"`` and ``"dir/x"`` but
                not ``"dirx"``).
            report_datetime (datetime | date | str | None): select the batch
                for this target date (``YYYY-MM-DD``). Defaults to the newest
                manifest available under the destination path.
            chunk_size (int): network read chunk size in bytes.
            max_decompressed_bytes (int | None): per-shard cap on decompressed
                gzip bytes.
            max_record_bytes (int | None): cap on the (decompressed) byte
                size of a single CSV record.
            head_missing_metadata (bool): when a required metadata field is
                absent from the report schema or empty for a record, fetch it
                with an explicit per-object HEAD request. When False (the
                default), such records fail instead. HEAD calls are only made
                for records that actually miss data, never unconditionally.
            head_concurrency (int): maximum number of concurrent HEAD calls
                used by the fallback above.
            record_queue_size (int): maximum number of parsed records buffered
                ahead of the consumer; bounds memory and the amount of work
                discarded on early exit.
            publish_dircache (bool): after the snapshot has been fully read
                without errors, atomically publish the full-bucket listing to
                ``gcs_file_system.dircache``. Only allowed for snapshot
                listing without a prefix. A partial/failed iteration never
                touches the directory cache.

        Returns:
            InventorySnapshot: async-iterable snapshot (also an async context
                manager). Iterating yields object dictionaries; batch metadata
                (``snapshot_time``, ``target_datetime``, ``shards``,
                ``records_processed``, ``schema``) is exposed as attributes.
        """
        cls._validate_inventory_report_info(inventory_report_info)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if max_record_bytes is not None and max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be a positive integer or None.")
        if max_decompressed_bytes is not None and max_decompressed_bytes <= 0:
            raise ValueError(
                "max_decompressed_bytes must be a positive integer or None."
            )
        if record_queue_size <= 0:
            raise ValueError("record_queue_size must be a positive integer.")
        if head_concurrency < 1:
            raise ValueError("head_concurrency must be at least 1.")

        use_snapshot_listing = inventory_report_info.get("use_snapshot_listing")
        location = inventory_report_info.get("location")
        report_id = inventory_report_info.get("id")

        if publish_dircache and (not use_snapshot_listing or prefix):
            raise ValueError(
                "publish_dircache is only supported for full-bucket snapshot "
                "listing (use_snapshot_listing=True) without a prefix, so a "
                "partial inventory can never replace the global directory cache."
            )

        raw_config = await cls._fetch_raw_inventory_report_config(
            gcs_file_system=gcs_file_system,
            location=location,
            id=report_id,
        )
        # The size requirement for snapshot listing is re-checked against the
        # pinned manifest below, together with the HEAD fallback policy, so the
        # streaming path reports one consistent integrity error.
        report_config = cls._parse_raw_inventory_report_config(
            raw_inventory_report_config=raw_config,
            use_snapshot_listing=use_snapshot_listing,
            require_size=False,
        )
        all_metadata = await cls._fetch_inventory_report_metadata(
            gcs_file_system=gcs_file_system,
            inventory_report_config=report_config,
        )

        batch, head_fields = await cls._select_report_batch(
            gcs_file_system=gcs_file_system,
            report_config=report_config,
            report_id=report_id,
            all_metadata=all_metadata,
            report_datetime=report_datetime,
            use_snapshot_listing=use_snapshot_listing,
            head_missing_metadata=head_missing_metadata,
        )

        return InventorySnapshot(
            gcs_file_system=gcs_file_system,
            batch=batch,
            prefix=prefix or "",
            use_snapshot_listing=use_snapshot_listing,
            chunk_size=chunk_size,
            max_decompressed_bytes=max_decompressed_bytes,
            max_record_bytes=max_record_bytes,
            head_fields=head_fields,
            head_missing_metadata=head_missing_metadata,
            head_concurrency=head_concurrency,
            record_queue_size=record_queue_size,
            publish_dircache=publish_dircache,
        )

    @classmethod
    async def _select_report_batch(
        cls,
        gcs_file_system,
        report_config,
        report_id,
        all_metadata,
        report_datetime,
        use_snapshot_listing,
        head_missing_metadata,
    ):
        """Pick one manifest, validate it and pin all shard generations."""
        manifests = [
            metadata
            for metadata in all_metadata
            if metadata.get("name", "").endswith(_MANIFEST_SUFFIX)
        ]
        if not manifests:
            raise InventoryReportIntegrityError(
                "No inventory report manifest (*_manifest.json) was found under "
                f"{report_config.destination_path!r}. The manifest is the only "
                "marker proving that every shard of a batch has been generated; "
                "without it the reader cannot distinguish a complete batch from "
                "a partial one, so no snapshot is produced."
            )

        chosen_metadata = cls._choose_manifest(manifests, report_datetime)
        manifest = await cls._download_manifest(
            gcs_file_system=gcs_file_system,
            bucket=report_config.bucket,
            manifest_metadata=chosen_metadata,
        )
        batch, head_fields = cls._validate_manifest_batch(
            manifest=manifest,
            manifest_metadata=chosen_metadata,
            all_metadata=all_metadata,
            report_config=report_config,
            report_id=report_id,
            use_snapshot_listing=use_snapshot_listing,
            head_missing_metadata=head_missing_metadata,
        )
        return batch, head_fields

    @classmethod
    def _choose_manifest(cls, manifests, report_datetime):
        """Select the manifest for ``report_datetime``, else the newest one."""
        if report_datetime is None:
            return sorted(
                manifests,
                key=lambda item: cls._convert_str_to_datetime(item.get("timeCreated")),
                reverse=True,
            )[0]

        if isinstance(report_datetime, datetime):
            target_date = report_datetime.date()
        elif isinstance(report_datetime, date):
            target_date = report_datetime
        else:
            target_date = datetime.strptime(report_datetime, "%Y-%m-%d").date()

        matches = []
        for metadata in manifests:
            base_name = metadata["name"].rsplit("/", 1)[-1]
            match = _MANIFEST_NAME_RE.match(base_name)
            if (
                match is not None
                and datetime.strptime(match.group("target_date"), "%Y-%m-%d").date()
                == target_date
            ):
                matches.append(metadata)
        if not matches:
            raise InventoryReportIntegrityError(
                f"No inventory report manifest found for target date {target_date}."
            )
        # Several manifests on the same day: take the latest generated one.
        return sorted(
            matches,
            key=lambda item: cls._convert_str_to_datetime(item.get("timeCreated")),
            reverse=True,
        )[0]

    @classmethod
    async def _download_manifest(cls, gcs_file_system, bucket, manifest_metadata):
        try:
            _headers, encoded_manifest = await gcs_file_system._call(
                "GET",
                "b/{}/o/{}",
                bucket,
                manifest_metadata.get("name"),
                alt="media",
            )
        except Exception as exc:
            raise InventoryReportIntegrityError(
                f"Failed to download inventory report manifest "
                f"{manifest_metadata.get('name')!r}: {exc}"
            ) from exc
        try:
            return json.loads(encoded_manifest)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InventoryReportIntegrityError(
                f"Inventory report manifest {manifest_metadata.get('name')!r} "
                f"does not contain valid JSON: {exc}"
            ) from exc

    @classmethod
    def _validate_manifest_batch(
        cls,
        manifest,
        manifest_metadata,
        all_metadata,
        report_config,
        report_id,
        use_snapshot_listing,
        head_missing_metadata,
    ):
        """Validate manifest content and pin each shard to its generation."""
        try:
            manifest_report_config = manifest["report_config"]
            shard_count = int(manifest["shard_count"])
            shard_file_names = list(manifest["report_shards_file_names"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InventoryReportIntegrityError(
                f"Manifest {manifest_metadata.get('name')!r} is missing a required "
                f"field (report_config, shard_count or report_shards_file_names): {exc}"
            ) from exc

        if shard_count <= 0:
            raise InventoryReportIntegrityError(
                f"Manifest {manifest_metadata.get('name')!r} declares no shards."
            )
        if len(shard_file_names) != shard_count:
            raise InventoryReportIntegrityError(
                f"Manifest {manifest_metadata.get('name')!r} declares "
                f"shard_count={shard_count} but lists "
                f"{len(shard_file_names)} shard file names."
            )
        if len(set(shard_file_names)) != len(shard_file_names):
            raise InventoryReportIntegrityError(
                f"Manifest {manifest_metadata.get('name')!r} lists duplicate "
                "shard file names."
            )

        # The manifest must belong to the report configuration the client asked
        # for, and its schema must match the live report config.
        config_name = manifest_report_config.get("name", "")
        if report_id and not config_name.endswith(f"reportConfigs/{report_id}"):
            raise InventoryReportIntegrityError(
                f"Manifest report config {config_name!r} does not match the "
                f"requested report config id {report_id!r}."
            )
        cls._validate_manifest_schema(
            manifest_report_config=manifest_report_config,
            report_config=report_config,
            use_snapshot_listing=use_snapshot_listing,
            head_missing_metadata=head_missing_metadata,
        )

        target_datetime = cls._manifest_target_date(manifest)
        snapshot_time = manifest.get("snapshot_time")

        # Index every object under the destination path by base name.
        metadata_by_base_name = {}
        for metadata in all_metadata:
            base_name = metadata.get("name", "").rsplit("/", 1)[-1]
            if base_name.endswith(_MANIFEST_SUFFIX):
                continue
            metadata_by_base_name.setdefault(base_name, []).append(metadata)

        pinned_shards = []
        shard_dates = set()
        for shard_index, shard_file_name in enumerate(shard_file_names):
            candidates = metadata_by_base_name.get(shard_file_name)
            if not candidates:
                raise InventoryReportIntegrityError(
                    f"Shard {shard_file_name!r} listed by manifest "
                    f"{manifest_metadata.get('name')!r} is missing from the "
                    "destination bucket; the batch is incomplete and no "
                    "snapshot can be produced."
                )
            if len(candidates) > 1:
                raise InventoryReportIntegrityError(
                    f"Shard name {shard_file_name!r} resolves to multiple "
                    "objects under the destination path."
                )
            shard_metadata = candidates[0]

            shard_date = cls._convert_str_to_datetime(
                shard_metadata["timeCreated"]
            ).date()
            shard_dates.add(shard_date)
            if target_datetime is not None and shard_date != target_datetime:
                raise InventoryReportIntegrityError(
                    f"Shard {shard_file_name!r} was created on {shard_date} but "
                    f"the selected batch targets {target_datetime}; shards from "
                    "different report times must not be mixed into one snapshot."
                )

            generation = shard_metadata.get("generation")
            if generation is None:
                raise InventoryReportIntegrityError(
                    f"Shard {shard_file_name!r} has no generation metadata; its "
                    "content version cannot be pinned."
                )
            pinned_shards.append(
                PinnedShard(
                    name=shard_metadata["name"],
                    generation=str(generation),
                    size=cls._optional_int(shard_metadata.get("size")),
                    compressed=cls._is_gzip_shard(shard_metadata, shard_file_name),
                    shard_index=shard_index,
                )
            )

        if len(shard_dates) > 1:
            raise InventoryReportIntegrityError(
                f"Shards of batch {manifest_metadata.get('name')!r} carry "
                f"different creation dates {sorted(shard_dates)}; refusing to "
                "mix shards from different report times."
            )

        records_processed = manifest.get("records_processed")
        if records_processed is not None:
            records_processed = int(records_processed)

        required_fields = ["name"] + (["size"] if use_snapshot_listing else [])
        head_fields = tuple(
            field_name
            for field_name in required_fields
            if field_name not in report_config.metadata_fields
        )
        if head_fields and not head_missing_metadata:
            raise InventoryReportIntegrityError(
                "The report schema is missing required metadata field(s) "
                f"{list(head_fields)}; enable head_missing_metadata to fetch "
                "them with explicit, concurrency-limited HEAD requests."
            )

        batch = PinnedReportBatch(
            source_bucket=cls._manifest_source_bucket(manifest_report_config),
            destination_bucket=report_config.bucket,
            manifest_name=manifest_metadata["name"],
            snapshot_time=snapshot_time,
            target_datetime=(
                target_datetime.isoformat() if target_datetime is not None else None
            ),
            records_processed=records_processed,
            shards=tuple(pinned_shards),
            csv_options=dict(report_config.csv_options or {}),
            metadata_fields=tuple(report_config.metadata_fields),
        )
        return batch, head_fields

    @classmethod
    def _validate_manifest_schema(
        cls,
        manifest_report_config,
        report_config,
        use_snapshot_listing,
        head_missing_metadata,
    ):
        """Cross-check csv options and metadata fields between configs."""
        manifest_options = (
            manifest_report_config.get("objectMetadataReportOptions", {}) or {}
        )
        manifest_fields = manifest_options.get("metadataFields")
        if not isinstance(manifest_fields, list) or "name" not in manifest_fields:
            raise InventoryReportIntegrityError(
                "The manifest report config does not include the required "
                f"'name' metadata field; got {manifest_fields!r}."
            )
        if list(manifest_fields) != list(report_config.metadata_fields):
            raise InventoryReportIntegrityError(
                "Metadata fields in the manifest report config "
                f"{manifest_fields!r} do not match the live report config "
                f"{report_config.metadata_fields!r}; schema drift detected."
            )
        if use_snapshot_listing and "size" not in manifest_fields:
            if not head_missing_metadata:
                raise InventoryReportIntegrityError(
                    "Snapshot listing requires the 'size' metadata field, which "
                    "is absent from the report schema; enable "
                    "head_missing_metadata to fetch it per object instead."
                )

        manifest_csv = manifest_report_config.get("csvOptions", {}) or {}
        for option, default in (
            ("recordSeparator", "\n"),
            ("delimiter", ","),
            ("headerRequired", False),
        ):
            manifest_value = manifest_csv.get(option, default)
            live_value = (report_config.csv_options or {}).get(option, default)
            if manifest_value != live_value:
                raise InventoryReportIntegrityError(
                    f"CSV option {option!r} in the manifest ({manifest_value!r}) "
                    f"does not match the live report config ({live_value!r})."
                )

    @classmethod
    def _manifest_target_date(cls, manifest):
        target = manifest.get("target_datetime")
        if isinstance(target, dict):
            try:
                return date(
                    year=int(target["year"]),
                    month=int(target["month"]),
                    day=int(target["day"]),
                )
            except (KeyError, TypeError, ValueError):
                return None
        if isinstance(target, str):
            try:
                return datetime.strptime(target, "%Y-%m-%d").date()
            except ValueError:
                return None
        return None

    @classmethod
    def _manifest_source_bucket(cls, manifest_report_config):
        options = manifest_report_config.get("objectMetadataReportOptions", {}) or {}
        filters = options.get("storageFilters", {}) or {}
        bucket = filters.get("bucket")
        if not bucket:
            raise InventoryReportIntegrityError(
                "Manifest report config does not identify the source bucket."
            )
        return bucket

    @staticmethod
    def _optional_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_gzip_shard(shard_metadata, shard_file_name):
        encoding = (shard_metadata.get("contentEncoding") or "").lower()
        content_type = (shard_metadata.get("contentType") or "").lower()
        return (
            encoding == "gzip"
            or content_type == "application/gzip"
            or shard_file_name.endswith(".gz")
        )

    @classmethod
    async def _iter_media_chunks(cls, gcs_file_system, bucket, shard, chunk_size):
        """Yield raw byte chunks of one pinned shard object.

        The pinned ``generation`` is sent on the request so the bytes returned
        are the exact shard version selected during batch validation. The
        response context is closed when the generator finishes or is closed
        (including consumer cancellation).
        """
        await gcs_file_system._set_session()
        url = gcs_file_system._format_path("b/{}/o/{}", (bucket, shard.name))
        params = gcs_file_system._get_params(
            {"alt": "media", "generation": shard.generation}
        )
        headers = gcs_file_system._get_headers(None, cache_source="inventory_report")
        async with gcs_file_system.session.request(
            "GET",
            url,
            params=params,
            headers=headers,
            timeout=gcs_file_system.requests_timeout,
        ) as response:
            if response.status >= 400:
                validate_response(response.status, await response.read(), url)
            async for chunk in response.content.iter_chunked(chunk_size):
                yield chunk

    @staticmethod
    def _key_matches_prefix(key, prefix):
        """Match ``prefix`` on object-key (``/`` segment) boundaries."""
        if not prefix:
            return True
        normalized = prefix.rstrip("/")
        return key == normalized or key.startswith(normalized + "/")


class InventoryReportConfig:
    """
    Represents the configuration for fetching inventory reports.

    Attributes:
        csv_options (dict): A dictionary containing options for parsing CSV
        format in the inventory reports.
        bucket (str): The name of the GCS bucket from which to fetch the
        inventory reports.
        destination_path (str): The path within the GCS bucket where the
        inventory reports are stored.
        metadata_fields (list): A list of strings representing metadata
        fields to be extracted from the inventory reports.
        obj_name_idx (int): The index of the "name" field in the 'metadata_fields'
        list, used to identify object names.
    """

    def __init__(
        self, csv_options, bucket, destination_path, metadata_fields, obj_name_idx
    ):
        self.csv_options = csv_options
        self.bucket = bucket
        self.destination_path = destination_path
        self.metadata_fields = metadata_fields
        self.obj_name_idx = obj_name_idx


class _GzipChunker:
    """Incremental gzip decompressor with a decompressed-byte budget.

    Fed with arbitrary compressed byte chunks; corruption (bad gzip header or
    truncated footer) is surfaced as :class:`InventoryReportError`, never as a
    silent truncation.
    """

    def __init__(self, max_decompressed_bytes, shard_name):
        # 16 + MAX_WBITS selects the gzip wrapper rather than raw/zlib frames.
        self._decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._max_decompressed_bytes = max_decompressed_bytes
        self._shard_name = shard_name
        self.bytes_out = 0
        self._input_seen = False

    def feed(self, compressed_chunk):
        if not compressed_chunk:
            return b""
        self._input_seen = True
        try:
            decompressed = self._decompressor.decompress(compressed_chunk)
        except zlib.error as exc:
            raise InventoryReportError(
                f"Corrupt gzip data in shard {self._shard_name!r}: {exc}"
            ) from exc
        self._account(decompressed)
        return decompressed

    def finish(self):
        if not self._input_seen:
            raise InventoryReportError(
                f"Gzip shard {self._shard_name!r} contained no data."
            )
        try:
            tail = self._decompressor.flush()
        except zlib.error as exc:
            raise InventoryReportError(
                f"Corrupt gzip footer in shard {self._shard_name!r}: {exc}"
            ) from exc
        if not self._decompressor.eof:
            raise InventoryReportError(
                f"Truncated gzip stream in shard {self._shard_name!r}: the "
                "end-of-stream marker was never reached."
            )
        self._account(tail)
        return tail

    def _account(self, decompressed):
        self.bytes_out += len(decompressed)
        if (
            self._max_decompressed_bytes is not None
            and self.bytes_out > self._max_decompressed_bytes
        ):
            raise InventoryReportLimitError(
                "Decompressed shard exceeds the configured byte budget",
                shard=self._shard_name,
                limit=self._max_decompressed_bytes,
                actual=self.bytes_out,
            )


class _StreamingCsvParser:
    """Incremental, RFC 4180 style CSV tokenizer over byte chunks.

    Delimiter, quote and record separator are all ASCII, so they can never
    collide with UTF-8 continuation bytes: tokenizing at byte level and
    decoding one fully assembled record at a time stays correct even when a
    multi-byte character is split across chunks. Quoted delimiters, quoted
    record separators (including embedded newlines) and doubled quote marks
    are handled by an explicit state machine instead of splitting on physical
    lines.
    """

    _QUOTE = 0x22  # '"'
    _CR = 0x0D
    _LF = 0x0A

    def __init__(
        self,
        *,
        delimiter,
        record_separator,
        header_required,
        expected_header,
        max_record_bytes,
        shard_name,
    ):
        self._delimiter = self._as_ascii_byte(delimiter, "delimiter")
        separator = record_separator.encode("ascii")
        if separator not in (b"\n", b"\r\n"):
            raise ValueError(
                "record_separator must be '\\n' or '\\r\\n', got "
                f"{record_separator!r}"
            )
        self._separator = separator
        self._header_required = header_required
        self._expected_header = expected_header
        self._max_record_bytes = max_record_bytes
        self._shard_name = shard_name

        self._fields = [bytearray()]
        self._record_touched = False
        self._quoted = False
        self._quote_closed = False
        self._pending_cr = False
        self._byte_offset = 0
        self._record_start_offset = 0
        self._record_bytes = 0
        self._header_consumed = False
        self._data_record_index = -1

    @staticmethod
    def _as_ascii_byte(value, label):
        if not isinstance(value, str) or len(value) != 1 or ord(value) > 127:
            raise ValueError(f"{label} must be a single ASCII character, got {value!r}")
        return ord(value)

    def feed(self, chunk):
        """Parse one byte chunk; return the records completed within it."""
        events = []
        for byte in chunk:
            self._byte_offset += 1

            # A CR may be the first half of a CRLF record separator.
            if self._pending_cr:
                self._pending_cr = False
                if byte == self._LF:
                    self._end_record(events)
                    continue
                # The CR was ordinary content; retain it and keep processing.
                self._append_byte(self._CR)

            if self._quoted:
                if byte == self._QUOTE:
                    self._quoted = False
                    self._quote_closed = True
                else:
                    self._append_byte(byte)
                continue

            if self._quote_closed:
                if byte == self._QUOTE:
                    # Doubled quote ("") inside a quoted field: literal quote.
                    self._fields[-1].append(self._QUOTE)
                    self._quote_closed = False
                    self._quoted = True
                elif byte == self._delimiter:
                    self._fields.append(bytearray())
                    self._quote_closed = False
                    self._record_touched = True
                elif self._separator == b"\r\n" and byte == self._CR:
                    self._pending_cr = True
                elif self._separator == b"\n" and byte == self._LF:
                    self._end_record(events)
                else:
                    raise InventoryReportParseError(
                        "Unexpected byte after the closing quote of a CSV field",
                        shard=self._shard_name,
                        record_index=max(self._data_record_index, 0),
                        byte_offset=self._byte_offset,
                    )
                continue

            if byte == self._QUOTE:
                if self._fields[-1]:
                    raise InventoryReportParseError(
                        "Quote character is only allowed at the start of a field",
                        shard=self._shard_name,
                        record_index=max(self._data_record_index, 0),
                        byte_offset=self._byte_offset,
                    )
                self._quoted = True
                self._record_touched = True
            elif byte == self._delimiter:
                self._fields.append(bytearray())
                self._record_touched = True
            elif self._separator == b"\r\n" and byte == self._CR:
                self._pending_cr = True
            elif self._separator == b"\n" and byte == self._LF:
                self._end_record(events)
            else:
                self._append_byte(byte)
                self._record_touched = True
        return events

    def finish(self):
        """Flush the final record after all shard chunks have been fed."""
        events = []
        if self._pending_cr:
            # A dangling CR at EOF is ordinary content, not a separator.
            self._pending_cr = False
            self._append_byte(self._CR)
        if self._record_touched:
            self._end_record(events)
        elif self._header_required and not self._header_consumed:
            raise InventoryReportParseError(
                "Header is required but the shard contained no record",
                shard=self._shard_name,
                byte_offset=self._byte_offset,
            )
        return events

    def _append_byte(self, byte):
        self._record_bytes += 1
        if (
            self._max_record_bytes is not None
            and self._record_bytes > self._max_record_bytes
        ):
            raise InventoryReportLimitError(
                "CSV record exceeds the configured single-record byte budget",
                shard=self._shard_name,
                record_index=max(self._data_record_index, 0),
                byte_offset=self._record_start_offset,
                limit=self._max_record_bytes,
                actual=self._record_bytes,
            )
        self._fields[-1].append(byte)

    def _end_record(self, events):
        if not self._record_touched:
            # Blank line / trailing separator: no empty record is emitted.
            self._reset_record()
            return
        decoded = self._decode_record()
        start_offset = self._record_start_offset
        record_bytes = self._record_bytes
        self._reset_record()

        if self._header_required and not self._header_consumed:
            self._header_consumed = True
            if self._expected_header is not None and decoded != list(
                self._expected_header
            ):
                raise InventoryReportParseError(
                    "CSV header does not match the report schema: "
                    f"expected {list(self._expected_header)}, got {decoded}",
                    shard=self._shard_name,
                    byte_offset=start_offset,
                )
            return

        self._data_record_index += 1
        events.append((decoded, self._data_record_index, start_offset, record_bytes))

    def _decode_record(self):
        if self._quoted:
            raise InventoryReportParseError(
                "Unterminated quoted field at end of record",
                shard=self._shard_name,
                record_index=max(self._data_record_index, 0),
                byte_offset=self._record_start_offset,
            )
        try:
            return [bytes(field).decode("utf-8") for field in self._fields]
        except UnicodeDecodeError as exc:
            raise InventoryReportParseError(
                f"Record is not valid UTF-8: {exc}",
                shard=self._shard_name,
                record_index=max(self._data_record_index, 0),
                byte_offset=self._record_start_offset,
            ) from exc

    def _reset_record(self):
        self._fields = [bytearray()]
        self._record_touched = False
        self._quoted = False
        self._quote_closed = False
        self._pending_cr = False
        self._record_bytes = 0
        self._record_start_offset = self._byte_offset


class _HeadMetadataFiller:
    """Bounded, on-demand HEAD fallback for records missing required fields.

    A HEAD request is only issued when a record actually misses a required
    value, and concurrent requests are limited by a semaphore -- there is no
    unconditional per-record request.
    """

    def __init__(self, gcs_file_system, bucket, fields, concurrency):
        self._gcs_file_system = gcs_file_system
        self._bucket = bucket
        self._fields = tuple(fields)
        self._semaphore = asyncio.Semaphore(concurrency)
        self.head_calls = 0
        self.max_in_flight = 0
        self._in_flight = 0

    async def fill(self, resource, key, missing_fields):
        self.head_calls += 1
        path = f"{self._bucket}/{key}"
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            async with self._semaphore:
                info = await self._gcs_file_system._get_object(path)
        finally:
            self._in_flight -= 1
        for field_name in missing_fields:
            if resource.get(field_name) in (None, "") and field_name in info:
                resource[field_name] = info[field_name]
        return resource


class _QueueEnd:
    """Sentinel: every shard was read and the snapshot is complete."""


class _QueueError:
    """Sentinel: the producer failed; the consumer re-raises the error."""

    def __init__(self, error):
        self.error = error


class InventorySnapshot:
    """Async iterable representation of one pinned inventory report batch.

    Returned by :meth:`InventoryReport.iter_snapshot`. Usable as an async
    context manager or directly with ``async for`` (``aclose`` is awaited
    automatically when the loop exits early). Only one shard response is open
    at a time and the parsed-record queue is bounded, so memory stays limited
    and early cancellation tears down the in-flight response, decompressor and
    producer task immediately.

    A snapshot is marked :attr:`completed` only after every pinned shard has
    been parsed and the manifest record count matches; partial iteration never
    publishes anything to the filesystem directory cache.
    """

    def __init__(
        self,
        gcs_file_system,
        batch,
        prefix,
        use_snapshot_listing,
        *,
        chunk_size,
        max_decompressed_bytes,
        max_record_bytes,
        head_fields,
        head_missing_metadata,
        head_concurrency,
        record_queue_size,
        publish_dircache,
    ):
        self._gcs_file_system = gcs_file_system
        self._batch = batch
        self._prefix = prefix
        self._use_snapshot_listing = use_snapshot_listing
        self._chunk_size = chunk_size
        self._max_decompressed_bytes = max_decompressed_bytes
        self._max_record_bytes = max_record_bytes
        self._head_fields = head_fields
        self._head_missing_metadata = head_missing_metadata
        self._head_concurrency = head_concurrency
        self._publish_dircache = publish_dircache

        self._queue = asyncio.Queue(maxsize=record_queue_size)
        self._producer = None
        self._closed = False
        self._filler = None
        self._collected = [] if publish_dircache else None

        # Public, read-only batch metadata.
        self.snapshot_time = batch.snapshot_time
        self.target_datetime = batch.target_datetime
        self.records_processed = batch.records_processed
        self.shards = batch.shards
        self.schema = batch.metadata_fields
        self.completed = False

    @property
    def manifest_name(self):
        return self._batch.manifest_name

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        if self._producer is None:
            self._producer = asyncio.ensure_future(self._produce())
        elif self._producer.done() and self._queue.empty():
            # The batch was fully consumed (or already failed) on a prior call.
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, _QueueEnd):
            self.completed = True
            raise StopAsyncIteration
        if isinstance(item, _QueueError):
            self.completed = False
            raise item.error
        return item

    async def aclose(self):
        """Cancel production and release the open response/decompressor.

        A snapshot that already finished naturally keeps ``completed=True``;
        only an actually interrupted run is marked incomplete.
        """
        if self._closed:
            return
        self._closed = True
        producer_running = self._producer is not None and not self._producer.done()
        if producer_running:
            self.completed = False
            self._producer.cancel()
        if self._producer is not None:
            await asyncio.gather(self._producer, return_exceptions=True)
        # Discard the bounded amount of buffered work; nothing else is retained.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._collected = None

    def __del__(self):
        # Safety net for consumers that abandon the iterator without
        # ``async with``/``aclose``: cancelling the task lets its response
        # context and decompressor unwind on the next loop iteration.
        producer = self.__dict__.get("_producer")
        if producer is not None and not producer.done():
            try:
                producer.cancel()
            except RuntimeError:
                pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.aclose()
        return False

    async def _produce(self):
        try:
            if self._head_missing_metadata:
                self._filler = _HeadMetadataFiller(
                    self._gcs_file_system,
                    self._batch.source_bucket,
                    self._head_fields,
                    self._head_concurrency,
                )
            total_records = 0
            for shard in self._batch.shards:
                shard_generator = self._produce_shard(shard)
                try:
                    async for obj, key in shard_generator:
                        total_records += 1
                        if not InventoryReport._key_matches_prefix(key, self._prefix):
                            continue
                        if self._collected is not None:
                            self._collected.append(obj)
                        await self._queue.put(obj)
                finally:
                    # Explicit aclose so a cancel landing at the queue put
                    # still unwinds the shard response and decompressor.
                    await shard_generator.aclose()

            if (
                self._batch.records_processed is not None
                and total_records != self._batch.records_processed
            ):
                raise InventoryReportIntegrityError(
                    f"Manifest {self._batch.manifest_name!r} claims "
                    f"{self._batch.records_processed} records but the pinned "
                    f"shards yielded {total_records}; the batch content does "
                    "not match its manifest."
                )

            if self._publish_dircache:
                # Atomic publication only after a fully verified snapshot.
                self._gcs_file_system.dircache[self._batch.source_bucket] = list(
                    self._collected
                )
            self.completed = True
            await self._queue.put(_QueueEnd())
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - re-raised via the queue
            self.completed = False
            await self._queue.put(_QueueError(exc))

    async def _produce_shard(self, shard):
        csv_options = self._batch.csv_options or {}
        parser = _StreamingCsvParser(
            delimiter=csv_options.get("delimiter", ","),
            record_separator=csv_options.get("recordSeparator", "\n"),
            header_required=csv_options.get("headerRequired", False),
            expected_header=self._batch.metadata_fields,
            max_record_bytes=self._max_record_bytes,
            shard_name=shard.name,
        )
        decompressor = (
            _GzipChunker(self._max_decompressed_bytes, shard.name)
            if shard.compressed
            else None
        )
        media_generator = InventoryReport._iter_media_chunks(
            self._gcs_file_system,
            self._batch.destination_bucket,
            shard,
            self._chunk_size,
        )
        try:
            async for raw_chunk in media_generator:
                data = (
                    decompressor.feed(raw_chunk)
                    if decompressor is not None
                    else raw_chunk
                )
                for event in parser.feed(data):
                    yield await self._map_record(shard, event)
            if decompressor is not None:
                for event in parser.feed(decompressor.finish()):
                    yield await self._map_record(shard, event)
            for event in parser.finish():
                yield await self._map_record(shard, event)
        except (InventoryReportError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise InventoryReportError(
                f"Unexpected failure while reading shard {shard.name!r}: {exc}"
            ) from exc
        finally:
            # Closes the network response (and releases the connection) even
            # when cancellation lands at a yield point.
            await media_generator.aclose()

    async def _map_record(self, shard, event):
        fields, record_index, byte_offset, _record_bytes = event
        schema = self._batch.metadata_fields
        if len(fields) != len(schema):
            raise InventoryReportParseError(
                f"Expected {len(schema)} fields per schema {list(schema)} but "
                f"the record has {len(fields)} fields",
                shard=shard.name,
                record_index=record_index,
                byte_offset=byte_offset,
            )

        resource = dict(zip(schema, fields))
        key = resource.get("name", "")
        if not key:
            raise InventoryReportParseError(
                "Record is missing the object 'name' field",
                shard=shard.name,
                record_index=record_index,
                byte_offset=byte_offset,
            )

        if self._use_snapshot_listing:
            await self._fill_required_fields(
                resource, key, shard, record_index, byte_offset
            )
            # Schema-driven conversions; unknown/extensible fields are kept.
            if "size" in resource:
                try:
                    resource["size"] = int(resource["size"])
                except (TypeError, ValueError) as exc:
                    raise InventoryReportParseError(
                        f"Invalid integer size {resource.get('size')!r}",
                        shard=shard.name,
                        record_index=record_index,
                        byte_offset=byte_offset,
                    ) from exc
            # Empty timestamp values would break timestamp parsing; drop them.
            for timestamp_field in ("updated", "timeCreated", "timeDeleted"):
                if timestamp_field in resource and not resource[timestamp_field]:
                    resource.pop(timestamp_field)
            obj = self._gcs_file_system._process_object(
                self._batch.source_bucket, resource
            )
        else:
            obj = {"name": f"{self._batch.source_bucket}/{key}"}
        return obj, key

    async def _fill_required_fields(
        self, resource, key, shard, record_index, byte_offset
    ):
        missing = [
            field_name
            for field_name in ("size",)
            if resource.get(field_name) in (None, "")
        ]
        # Fields absent from the schema entirely are always missing.
        missing.extend(
            field_name for field_name in self._head_fields if field_name not in missing
        )
        if not missing:
            return
        if not self._head_missing_metadata or self._filler is None:
            raise InventoryReportParseError(
                f"Record is missing required field(s) {missing} and "
                "head_missing_metadata is disabled",
                shard=shard.name,
                record_index=record_index,
                byte_offset=byte_offset,
            )
        # Only records that actually miss data trigger a HEAD request, and the
        # filler's semaphore bounds how many run concurrently.
        await self._filler.fill(resource, key, missing)
