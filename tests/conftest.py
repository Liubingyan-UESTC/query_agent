"""公共 pytest 配置。

仓库根目录通过 pyproject 的 `pythonpath = ["."]` 进入 sys.path，此处无需重复处理。
共享 fixture（runtime_with_mocks / fake_es / frozen_clock 等）在步骤 32 体系化收口。
"""
