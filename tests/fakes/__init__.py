"""测试替身包（§18.1）。

这里放的是**不触网、不依赖 SDK** 的假实现，供 ``tests/unit``、``tests/integration``
与 ``tests/e2e`` 共用：

* :mod:`tests.fakes.provider` —— ``FakeProvider``（实现 §13.1.3 的 ``ModelProvider``
  协议，**不 import openai**，因此 §5–§12 的所有模块都能在不装 SDK 的环境下测试）。
"""

from __future__ import annotations

__all__: list[str] = []
