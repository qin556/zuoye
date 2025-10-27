# 【第一步：必须放在文件最顶部】配置项目根目录路径
import sys
from pathlib import Path

# 1. 定位项目根目录（包含 app/ 和 tests/ 的目录）
current_test_file = Path(__file__).absolute()  # 当前 test_ml.py 的绝对路径
tests_directory = current_test_file.parent  # 上级目录：tests/
project_root = (
    tests_directory.parent
)  # 上上级目录：项目根目录（如 /home/runner/work/zuoye/zuoye/）

# 2. 强制将根目录加入 Python 搜索路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    # 可选：CI 中打印路径，方便调试（本地可注释）
    print(f"✅ 项目根目录已加入 sys.path：{project_root}")
else:
    print(f"✅ 项目根目录已在 sys.path 中：{project_root}")


# 【第二步：路径配置完成后，再导入 app 相关模块】
# 注意：删除不存在的导入（如 load_data，若 model.py 中没有该函数）
from app.model import train_model  # 仅导入 model.py 中实际存在的函数
from app.data import load_local_fashion_mnist  # 导入 data.py 中的函数
import pytest


# 【第三步：测试用例（根据你的实际功能调整）】
def test_load_local_fashion_mnist():
    """测试 Fashion MNIST 数据加载是否正常"""
    try:
        X_train, X_test, y_train, y_test, scaler = load_local_fashion_mnist()
        # 验证数据维度（Fashion MNIST 固定维度）
        assert X_train.shape == (
            60000,
            784,
        ), f"训练集维度错误，实际：{X_train.shape}"
        assert y_train.shape == (60000,), f"训练标签维度错误，实际：{y_train.shape}"
        assert len(set(y_train)) == 10, f"分类数错误，实际：{len(set(y_train))}"
        print("✅ 数据加载测试通过")
    except Exception as e:
        pytest.fail(f"数据加载测试失败：{str(e)}")


def test_train_model():
    """测试模型训练是否正常（短迭代快速验证）"""
    try:
        # 用小迭代次数快速测试（避免 CI 运行时间过长）
        model, test_acc = train_model(learning_rate=0.1, max_iter=200)
        # 验证准确率在合理范围（即使迭代少，准确率也应 >0.7）
        assert test_acc > 0.7, f"模型准确率异常，实际：{test_acc}"
        print(f"✅ 模型训练测试通过，测试准确率：{test_acc:.4f}")
    except Exception as e:
        pytest.fail(f"模型训练测试失败：{str(e)}")


# 可选：添加更多测试用例...
