from urllib import parse


def convert_to_fuse(input_path: str) -> str:
    """
    Path converter

    :param input_path:
        can be a GCS url
    :return:
        converted url

    Use this util to map buckets to local paths
    Buckets are mounted as subfolders of /gcs/
    See: https://cloud.google.com/vertex-ai/docs/training/cloud-storage-file-system

    Example: file gs://my_bucket/aa/bb is accessed through /gcs/my_bucket/aa/bb

    """
    parsed = parse.urlparse(input_path)

    if parsed.scheme == "gs":
        converted = "/gcs/" + parsed.netloc + parsed.path
    else:
        converted = input_path

    return converted
