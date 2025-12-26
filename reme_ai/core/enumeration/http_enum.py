from enum import Enum


class HttpEnum(str, Enum):
    GET = "get"

    POST = "post"

    HEAD = "head"

    PUT = "put"

    DELETE = "delete"
