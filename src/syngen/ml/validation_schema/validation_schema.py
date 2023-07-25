from typing import Dict
import json
from dataclasses import dataclass
from marshmallow import Schema, fields, validate, ValidationError, validates_schema
from loguru import logger


class ReferenceSchema(Schema):
    table = fields.String()
    columns = fields.List(fields.String())


class CaseInsensitiveString(fields.String):
    def _deserialize(self, value, attr, data, **kwargs):
        if not isinstance(value, str):
            raise ValidationError("Not a valid string.")
        return value.lower()


class KeysSchema(Schema):
    type = fields.String(validate=validate.OneOf(["FK", "PK", "UQ"]), required=True)
    columns = fields.List(fields.String(), required=True)
    joined_sample = fields.Boolean(required=False)
    references = fields.Nested(ReferenceSchema, required=False)

    @validates_schema
    def validate_references(self, data, **kwargs):
        if data["type"] == "FK" and "references" not in data:
            raise ValidationError("The 'references' field is required when 'type' is 'FK'")
        if data["type"] != "FK" and "references" in data:
            raise ValidationError("The 'references' field is only allowed when 'type' is 'FK'")


class TrainingSettingsSchema(Schema):
    epochs = fields.Integer(validate=validate.Range(min=1), required=False)
    drop_null = fields.Boolean(required=False)
    row_limit = fields.Integer(validate=validate.Range(min=1), allow_none=True, required=False)
    batch_size = fields.Integer(validate=validate.Range(min=1), required=False)
    print_report = fields.Boolean(required=False)


class ExtendedTrainingSettingsSchema(TrainingSettingsSchema):
    source = fields.String(required=True, allow_none=False)
    column_types = fields.Dict(
        keys=fields.String(validate=validate.OneOf(["categorical"])),
        values=fields.List(fields.String())
    )


class InferSettingsSchema(Schema):
    destination = fields.String(required=False)
    size = fields.Integer(validate=validate.Range(min=1), required=False)
    run_parallel = fields.Boolean(required=False)
    batch_size = fields.Integer(validate=validate.Range(min=1), allow_none=True, required=False)
    random_seed = fields.Integer(validate=validate.Range(min=0), allow_none=True, required=False)
    print_report = fields.Boolean(required=False)


class FormatSettingsSchema(Schema):
    sep = fields.String(required=False, allow_none=True)
    quotechar = fields.String(required=False, validate=validate.Length(equal=1))
    quoting = CaseInsensitiveString(required=False, validate=validate.OneOf(["minimal", "all", "non-numeric", "none"]))
    escapechar = fields.String(required=False, validate=validate.Length(equal=1))
    encoding = fields.String(required=False)
    header = fields.Raw(
        required=False,
        allow_none=True,
        validate=lambda x: isinstance(x, int) \
                           or x is None \
                           or (isinstance(x, str) and x == 'infer') \
                           or (isinstance(x, list) and all(isinstance(elem, int) for elem in x))
    )
    skiprows = fields.Raw(
        required=False,
        allow_none=True,
        validate=lambda x: isinstance(x, int) \
                           or x is None \
                           or (isinstance(x, list) and all(isinstance(elem, int) for elem in x)))
    on_bad_lines = CaseInsensitiveString(required=False, validate=validate.OneOf(["error", "warn", "skip"]))
    engine = fields.String(required=False, validate=validate.OneOf(["c", "python"]))


class GlobalSettingsSchema(Schema):
    train_settings = fields.Nested(TrainingSettingsSchema, required=False, allow_none=True)
    infer_settings = fields.Nested(InferSettingsSchema, required=False, allow_none=True)


class ConfigurationSchema(Schema):
    train_settings = fields.Nested(ExtendedTrainingSettingsSchema, required=True, allow_none=False)
    infer_settings = fields.Nested(InferSettingsSchema, required=False, allow_none=True)
    format = fields.Nested(FormatSettingsSchema, required=False, allow_none=True)
    keys = fields.Dict(keys=fields.String(), values=fields.Nested(KeysSchema), required=False, allow_none=True)


@dataclass
class ValidationSchema:
    metadata: Dict
    global_schema = GlobalSettingsSchema()
    configuration_schema = ConfigurationSchema()

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
            logger.error("Validation error(s) found in the metadata")
            for section, errors_details in errors.items():
                logger.error(f"The error(s) found in - \"{section}\": {json.dumps(errors_details, indent=4)}")
            raise ValidationError(f"Validation error(s) found in the metadata. The details are - {errors}")
        if not errors:
            logger.info("The metadata file is valid")
