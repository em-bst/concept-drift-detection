"""
List of bucket rows.

Add new bucket row at head of window, remove old bucket row from tail of window.
"""

# For ADWIN parts 
# Authors: blablahaha
#          Alexey Egorov
#          Yuqing Wei 
# Github link : https://github.com/blablahaha/concept-drift

from concept_drift.adwin_bucket_row import AdwinBucketRow, Adwin2BucketRow

class AdwinRowBucketList:
    def __init__(self, max_buckets=5):
        """
        :param max_buckets: Max number of bucket in each bucket row
        """
        self.max_buckets = max_buckets

        self.count = 0
        self.head = None
        self.tail = None
        self.__add_to_head()

    def __add_to_head(self):
        """
        Init bucket row list.
        """
        self.head = AdwinBucketRow(self.max_buckets, next_bucket_row=self.head)
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self):
        """
        Add the bucket row at the end of the window.
        """
        self.tail = AdwinBucketRow(self.max_buckets, previous_bucket_row=self.tail)
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self):
        """
        Remove the last bucket row in the window.
        """
        self.tail = self.tail.previous_bucket_row
        if self.tail is None:
            self.head = None
        else:
            self.tail.next_bucket_row = None
        self.count -= 1

"""
Optimized bucket list for ADWIN 2.
Main changes : 
- improving dynamic row management with the use of double-cahined lists
- allowing more efficient insertion and deletion
- to facilitate the merging of buckets and improve memory management
"""

class Adwin2RowBucketList:
    def __init__(self, max_buckets=5):
        """
        :param max_buckets: Max number of buckets in each bucket row
        """
        self.max_buckets = max_buckets

        self.count = 0
        self.head = None
        self.tail = None
        self.__add_to_head()

    def __add_to_head(self):
        """
        Add a new bucket row to the head of the list.
        """
        self.head = Adwin2BucketRow(self.max_buckets, next_bucket_row=self.head)
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self):
        """
        Add a new bucket row to the end of the list.
        """
        self.tail = Adwin2BucketRow(self.max_buckets, previous_bucket_row=self.tail)
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_tail(self):
        """
        Remove the last bucket row in the list.
        """
        if self.tail is not None:
            self.tail = self.tail.previous_bucket_row
            if self.tail is None:
                self.head = None
            else:
                self.tail.next_bucket_row = None
            self.count -= 1
