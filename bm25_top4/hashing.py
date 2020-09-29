import hashlib

def get_md5_hash(v):
    return hashlib.md5(v.encode()).hexdigest()
