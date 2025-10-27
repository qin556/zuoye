# 第一步：添加项目根目录到 Python 搜索路径（解决 ModuleNotFoundError）
import sys
from pathlib import Path

# 获取当前测试文件（test_ml.py）的路径，向上回溯到项目根目录
test_file_path = Path(__file__).resolve()  # 当前文件绝对路径
tests_dir = test_file_path.parent  # 上级目录：tests/
project_root = tests_dir.parent  # 上上级目录：项目根目录（包含 app/）

# 将项目根目录加入 sys.path（优先搜索）
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 第二步：正常导入所需模块（此时 app 已能被识别）
from app.model import train_model  # 根据实际需要调整导入内容
from app.data import load_local_fashion_mnist
import pytest


# 后续测试用例代码...
# 示例测试用例（根据你的实际功能调整）
def test_data_loading():
    # 测试数据加载是否正常
    X_train, X_test, y_train, y_test, scaler = load_local_fashion_mnist()
    assert X_train.shape == (60000, 784), "训练集维度错误"
    assert len(set(y_train)) == 10, "分类数错误（应为10分类）"


def test_model_training():
    # 测试模型训练是否正常
    model, test_acc = train_model(learning_rate=0.1, max_iter=1000)
    assert 0.85 <= test_acc <= 0.95, "测试准确率异常（正常范围0.85-0.95）"
