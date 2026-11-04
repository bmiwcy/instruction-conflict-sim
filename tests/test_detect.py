from ics.detect import is_chinese

def test_cases():
    assert is_chinese("") is None
    assert is_chinese("好的，下面是我的回答。")
    assert not is_chinese("Sure, here is the answer.")
    assert not is_chinese("はい、こちらが答えです。")          # Japanese != Chinese
    assert is_chinese("可以用 `dict` 实现：\n```python\nd = {}\nfor x in xs:\n    d[x] = 1\n```\n这样查找是 O(1)。")
    assert is_chinese("REST API 是一种基于 HTTP 的接口设计风格，常用于前后端通信。")
    assert not is_chinese("Here's how: 首先 you need to boil the water and then add rice.")

if __name__ == "__main__":
    test_cases(); print("ok")
