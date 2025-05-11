"""
Within one bucket row, each bucket contains same number of original data.

New buckets are added at the end of bucket row.
When old buckets need to be removed, they are taken from the head of this bucket row.
"""

# For ADWIN parts 
# Authors: blablahaha
#          Alexey Egorov
#          Yuqing Wei 
# Github link : https://github.com/blablahaha/concept-drift

import numpy as np


class AdwinBucketRow:
    def __init__(self, max_buckets=5, next_bucket_row=None, previous_bucket_row=None):
        """
        :param max_buckets: Max bucket with one row
        :param next_bucket_row: Following bucket row
        :param previous_bucket_row: Previous bucket row
        """
        self.max_buckets = max_buckets

        # Current count of buckets in this bucket row
        self.bucket_count = 0

        self.next_bucket_row = next_bucket_row
        # Set previous bucket row connect to this bucket row
        if next_bucket_row is not None:
            next_bucket_row.previous_bucket_row = self

        self.previous_bucket_row = previous_bucket_row
        # Set next bucket connect to this bucket
        if previous_bucket_row is not None:
            previous_bucket_row.next_bucket_row = self

        # Init statistic number for each bucket
        self.bucket_sum = np.zeros(self.max_buckets + 1)
        self.bucket_variance = np.zeros(self.max_buckets + 1)

    def insert_bucket(self, value, variance):
        """
        Insert a new bucket at the end.
        """
        self.bucket_sum[self.bucket_count] = value
        self.bucket_variance[self.bucket_count] = variance
        self.bucket_count += 1

    def compress_bucket(self, number_deleted):
        """
        Remove the oldest buckets.
        """
        delete_index = self.max_buckets - number_deleted + 1
        self.bucket_sum[:delete_index] = self.bucket_sum[number_deleted:]
        self.bucket_sum[delete_index:] = np.zeros(number_deleted)

        self.bucket_variance[:delete_index] = self.bucket_variance[number_deleted:]
        self.bucket_variance[delete_index:] = np.zeros(number_deleted)

        self.bucket_count -= number_deleted

"""
Optimized bucket row for ADWIN 2.

Main changes : 
- Optimization of the management of bucket rows
- simplification of the methods of adding (insert_bucket) and compressing (compress_bucket)
- just affecting the infrastructure of buckets
"""

class Adwin2BucketRow:
    def __init__(self, max_buckets=5, next_bucket_row=None, previous_bucket_row=None):
        """
        :param max_buckets: Max bucket with one row
        :param next_bucket_row: Following bucket row
        :param previous_bucket_row: Previous bucket row
        """
        self.max_buckets = max_buckets

        # Current count of buckets in this bucket row
        self.bucket_count = 0

        self.next_bucket_row = next_bucket_row
        if next_bucket_row is not None:
            next_bucket_row.previous_bucket_row = self

        self.previous_bucket_row = previous_bucket_row
        if previous_bucket_row is not None:
            previous_bucket_row.next_bucket_row = self

        # Initialize bucket statistics (sum and variance)
        self.bucket_sum = np.zeros(self.max_buckets)
        self.bucket_variance = np.zeros(self.max_buckets)

    def insert_bucket(self, value, variance):
        """
        Insert a new bucket at the end.
        """
        if self.bucket_count < self.max_buckets:
            self.bucket_sum[self.bucket_count] = value
            self.bucket_variance[self.bucket_count] = variance
            self.bucket_count += 1
        else:
            raise ValueError("Cannot insert bucket: maximum bucket count reached.")

    def compress_bucket(self, number_deleted):
        """
        Remove the oldest buckets.
        """
        if number_deleted >= self.bucket_count:
            self.bucket_sum.fill(0)
            self.bucket_variance.fill(0)
            self.bucket_count = 0
        else:
            self.bucket_sum[:-number_deleted] = self.bucket_sum[number_deleted:]
            self.bucket_sum[-number_deleted:] = 0

            self.bucket_variance[:-number_deleted] = self.bucket_variance[number_deleted:]
            self.bucket_variance[-number_deleted:] = 0

            self.bucket_count -= number_deleted
