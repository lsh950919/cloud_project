import os
import sys
import boto3
import argparse
import threading
from boto3.s3.transfer import TransferConfig


class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

GB = 1024 ** 3
config = TransferConfig(multipart_threshold=GB, max_concurrency=8)

'''
AKIAWG76QTMBBZCBJWIY FTKxfyo2yCvPYeauE2FHot5qAVTj9IitEhdE2wOj
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_location')
    parser.add_argument('file_name')
    args = parser.parse_args()
    client = boto3.client('s3', 
                        #   aws_access_key_id = 'AKIAWMMOC2MEVIUJGVCT', 
                        #   aws_secret_access_key = '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'
                          aws_access_key_id = 'AKIAWG76QTMBBZCBJWIY', 
                          aws_secret_access_key = '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'
                          )
    client.upload_file(args.file_location, "blim-extra", args.file_name, Callback = ProgressPercentage(args.file_location), Config = config)