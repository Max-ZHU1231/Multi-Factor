"""
factor_framework.data
=====================
阶段二数据存储抽象层。

模块
----
store.py : DataStore（抽象基类）+ CSVDataStore（具体实现）
"""

from factor_framework.data.store import DataStore, CSVDataStore

__all__ = ["DataStore", "CSVDataStore"]
