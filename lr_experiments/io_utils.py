from sqlalchemy import create_engine
from sqlalchemy.engine import URL 
from sqlalchemy.schema import CreateSchema

import yaml


def to_yaml(data, filename: str, parent_dir: str=".") -> dict:
    with open(f"{parent_dir}/{filename}", "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, sort_keys=True)


def from_yaml(filename: str, parent_dir: str=".") -> dict:
    with open(f"{parent_dir}/{filename}", "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def create_db_engine(configs: dict, schema=None):
    url = URL.create(
        drivername = "postgresql",
        username = configs["user"],
        password = configs["password"],
        host = configs["host"],
        port = configs["port"],
        database = configs["database"],
    )
    
    engine = create_engine(url)

    if schema is None and configs.get("schema"):
       engine.execute(CreateSchema(configs["schema"]))
    elif schema is not None:
        engine.execute(CreateSchema(schema))
    return engine