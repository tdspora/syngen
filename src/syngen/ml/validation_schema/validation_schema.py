from typing import Dict, Literal, List
import json
from pathlib import Path
from dataclasses import dataclass, field

from marshmallow import (
    Schema,
    fields,
    validate,
    ValidationError,
    validates_schema,
    post_load,
)
from loguru import logger

SUPPORTED_EXCEL_EXTENSIONS = [".xls", ".xlsx"]


@dataclass
class ReportTypes:
    infer_report_types: List[str] = field(default_factory=lambda: ["accuracy", "metrics_only"])
    train_report_types: List[str] = field(init=False)
    excluded_reports: List[str] = field(default_factory=lambda: ["metrics_only"])
    full_list_of_train_report_types: List[str] = field(init=False)
    full_list_of_infer_report_types: List[str] = field(init=False)

    def __post_init__(self):
        self.train_report_types = self.infer_report_types + ["sample"]
        self.full_list_of_train_report_types = self.get_list_of_report_types("train")
        self.full_list_of_infer_report_types = self.get_list_of_report_types("infer")

    def get_list_of_report_types(self, report_type):
        """
        Get the full list of reports that should be generated
        if the parameter 'reports' sets to 'all'
        """
        report_types = (
            self.train_report_types
            if report_type == "train"
            else self.infer_report_types
        )
        return [
            report
            for report in report_types
            if report not in self.excluded_reports
        ]


class ReferenceSchema(Schema):
    table = fields.String(required=True, allow_none=False)
    columns = fields.List(fields.String(), required=True, allow_none=False)


class CaseInsensitiveString(fields.String):
    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, str):
            raise ValidationError("Not a valid string.")
        return value.lower()


class KeysSchema(Schema):
    pk_types = ["PK", "UQ"]
    fk_types = ["FK"]
    type_of_keys = [*pk_types, *fk_types]
    type = fields.String(validate=validate.OneOf(type_of_keys), required=True)
    columns = fields.List(fields.String(), required=True, allow_none=False)
    references = fields.Nested(ReferenceSchema, required=False, allow_none=False)

    @validates_schema
    def validate_references(self, data, **kwargs):
        if not data["columns"]:
            raise ValidationError("The 'columns' field must not be empty")
        if data["type"] in self.fk_types and "references" not in data:
            raise ValidationError(
                f"The 'references' field is required when 'type' is "
                f"{' or '.join([f'{fk_type!r}' for fk_type in self.fk_types])}"
            )
        if data["type"] in self.fk_types and "references" in data and not data["references"]:
            raise ValidationError(
                f"The 'references' field must not be empty when 'type' is "
                f"{' or '.join([f'{fk_type!r}' for fk_type in self.fk_types])}"
            )
        if data["type"] not in self.fk_types and "references" in data:
            raise ValidationError(
                f"The 'references' field is only allowed when 'type' is "
                f"{' or '.join([f'{fk_type!r}' for fk_type in self.fk_types])}"
            )
        if data["type"] in self.fk_types and not data["references"]["columns"]:
            raise ValidationError("The 'references.columns' field must not be empty")
        if len(data["columns"]) != len(set(data["columns"])):
            raise ValidationError("The 'columns' field must contain unique values")
        if data["type"] in self.fk_types and len(data["references"]["columns"]) != len(
            set(data["references"]["columns"])
        ):
            raise ValidationError("The 'references.columns' field must contain unique values")
        if data["type"] in self.fk_types and len(data["columns"]) != len(
            data["references"].get("columns", [])
        ):
            raise ValidationError(
                "The 'columns' field must have the same length as 'references.columns'"
            )


class EncryptionSettings(Schema):
    fernet_key = fields.String(required=False, allow_none=True)


class TrainingSettingsSchema(Schema):
    @staticmethod
    def validate_reports(x):
        if any([i in ["all", "none"] for i in x]):
            raise ValidationError(
                "The value 'all' or 'none' might not be passed in the list."
            )
        if not (
            isinstance(x, list)
            and all(
                isinstance(elem, str)
                and elem in ReportTypes().train_report_types for elem in x
            )
        ):
            raise ValidationError("Invalid value.")

    epochs = fields.Integer(validate=validate.Range(min=1), required=False)
    drop_null = fields.Boolean(required=False)
    row_limit = fields.Integer(validate=validate.Range(min=1), allow_none=True, required=False)
    batch_size = fields.Integer(validate=validate.Range(min=1), required=False)
    reports = fields.Raw(
        required=False,
        validate=validate_reports
    )


class ExtendedRestrictedTrainingSettingsSchema(TrainingSettingsSchema):
    source = fields.String(required=True, allow_none=False)
    column_types = fields.Dict(
        keys=fields.String(validate=validate.OneOf(["categorical"])),
        values=fields.List(fields.String()),
    )


class ExtendedTrainingSettingsSchema(ExtendedRestrictedTrainingSettingsSchema):
    source = fields.String(required=False, allow_none=True)


class InferSettingsSchema(Schema):
    @staticmethod
    def validate_reports(x):
        if any([i in ["all", "none"] for i in x]):
            raise ValidationError(
                "The value 'all' or 'none' might not be passed in the list."
            )
        if not (
            isinstance(x, list)
            and all(
                isinstance(elem, str)
                and elem in ReportTypes().infer_report_types for elem in x
            )
        ):
            raise ValidationError("Invalid value.")

    destination = fields.String(required=False)
    size = fields.Integer(validate=validate.Range(min=1), required=False)
    run_parallel = fields.Boolean(required=False)
    batch_size = fields.Integer(validate=validate.Range(min=1), allow_none=True, required=False)
    random_seed = fields.Integer(validate=validate.Range(min=0), allow_none=True, required=False)
    reports = fields.Raw(
        required=False,
        validate=validate_reports
    )


class CSVFormatSettingsSchema(Schema):
    sep = fields.String(required=False, allow_none=True)
    quotechar = fields.String(required=False, validate=validate.Length(equal=1))
    quoting = CaseInsensitiveString(
        required=False,
        validate=validate.OneOf(["minimal", "all", "non-numeric", "none"]),
    )
    escapechar = fields.String(required=False, validate=validate.Length(equal=1))
    encoding = fields.String(required=False, allow_none=True)
    header = fields.Raw(
        required=False,
        allow_none=True,
        validate=lambda x: isinstance(x, int)
        or (isinstance(x, str) and x == "infer")
        or (isinstance(x, list) and all(isinstance(elem, int) for elem in x)),
    )
    skiprows = fields.Raw(
        required=False,
        allow_none=True,
        validate=lambda x: isinstance(x, int)
        or (isinstance(x, list) and all(isinstance(elem, int) for elem in x)),
    )
    on_bad_lines = CaseInsensitiveString(
        required=False, validate=validate.OneOf(["error", "warn", "skip"])
    )
    engine = fields.String(
        required=False, allow_none=True, validate=validate.OneOf(["c", "python"])
    )
    na_values = fields.List(fields.String(), required=False, allow_none=True)


class ExcelFormatSettingsSchema(Schema):
    sheet_name = fields.Raw(
        required=False,
        allow_none=True,
        validate=lambda x: isinstance(x, int)
        or x is None
        or isinstance(x, str)
        or (isinstance(x, list) and all(isinstance(elem, (int, str)) for elem in x)),
    )


class GlobalSettingsSchema(Schema):
    train_settings = fields.Nested(TrainingSettingsSchema, required=False, allow_none=True)
    infer_settings = fields.Nested(InferSettingsSchema, required=False, allow_none=True)
    encryption = fields.Nested(EncryptionSettings, required=False, allow_none=True)


class ConfigurationSchema(Schema):
    train_settings = fields.Nested(
        ExtendedTrainingSettingsSchema,
        required=False,
        allow_none=True
    )
    infer_settings = fields.Nested(InferSettingsSchema, required=False, allow_none=True)
    encryption = fields.Nested(EncryptionSettings, required=False, allow_none=True)
    format = fields.Raw(required=False, allow_none=True)
    keys = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(KeysSchema),
        required=False,
        allow_none=True,
    )

    @staticmethod
    def get_format_schema(source):
        if Path(source).suffix == ".csv":
            return CSVFormatSettingsSchema
        if Path(source).suffix in SUPPORTED_EXCEL_EXTENSIONS:
            return ExcelFormatSettingsSchema

    @post_load
    def process_format_field(self, data, **kwargs):
        train_settings = data.get("train_settings", {})
        path_to_source = train_settings.get("source") if train_settings else None
        if train_settings and path_to_source:
            format_schema = self.get_format_schema(path_to_source)
            if format_schema is not None and data.get("format") is not None:
                data["format"] = format_schema().load(data["format"])
        return data


class RestrictedConfigurationSchema(ConfigurationSchema):
    train_settings = fields.Nested(
        ExtendedRestrictedTrainingSettingsSchema,
        required=True,
        allow_none=False
    )


class ValidationSchema:
    def __init__(
        self,
        metadata: Dict,
        metadata_path: str,
        validation_source: bool,
        process: Literal["train", "infer"]
    ):
        self.metadata = metadata
        self.metadata_path = metadata_path
        self.global_schema = GlobalSettingsSchema()
        self.configuration_schema = (
            RestrictedConfigurationSchema()
            if validation_source and process == "train"
            else ConfigurationSchema()
        )

    def validate_schema(self):
        """
        Validate the metadata file
        """
        errors = {}
        for table_name in self.metadata.keys():
            try:
                if table_name == "global":
                    self.global_schema.load(self.metadata[table_name])
                else:
                    self.configuration_schema.load(self.metadata[table_name])
            except ValidationError as err:
                errors[table_name] = err.messages
        if errors:
            message = "Validation error(s) found in the schema of the metadata"
            logger.error(message)
            for section, errors_details in errors.items():
                logger.error(
                    f'The error(s) found in - "{section}": {json.dumps(errors_details, indent=4)}'
                )
            raise ValidationError(f"{message}. The details are - {errors}")
        if not errors:
            logger.debug("The schema of the metadata is valid")
