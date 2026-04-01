#!/usr/bin/env python3
"""
Ubuntu环境测试脚本

用于验证DAFA联邦学习框架在Ubuntu系统中的功能完整性和稳定性。

使用方法:
    python tests/test_ubuntu_environment.py
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest


pytestmark = pytest.mark.skip(
    reason="Environment verification script; run with `python tests/test_ubuntu_environment.py` instead of pytest."
)

IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, name: str, message: str = ""):
        self.passed.append((name, message))
        print(f"  [PASS] {name}" + (f" - {message}" if message else ""))
    
    def add_fail(self, name: str, message: str = ""):
        self.failed.append((name, message))
        print(f"  [FAIL] {name}" + (f" - {message}" if message else ""))
    
    def add_warning(self, name: str, message: str = ""):
        self.warnings.append((name, message))
        print(f"  [WARN] {name}" + (f" - {message}" if message else ""))
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 60)
        print("测试摘要")
        print("=" * 60)
        print(f"通过: {len(self.passed)}/{total}")
        print(f"失败: {len(self.failed)}/{total}")
        print(f"警告: {len(self.warnings)}")
        
        if self.failed:
            print("\n失败的测试:")
            for name, msg in self.failed:
                print(f"  - {name}: {msg}")
            return False
        return True


def test_system_info(result: TestResult):
    """测试系统信息"""
    print("\n[1] 系统信息测试")
    
    result.add_pass("操作系统", f"{platform.system()} {platform.release()}")
    result.add_pass("Python版本", platform.python_version())
    result.add_pass("架构", platform.machine())
    
    if IS_LINUX:
        try:
            with open("/etc/os-release") as f:
                content = f.read()
                for line in content.split("\n"):
                    if line.startswith("PRETTY_NAME="):
                        distro = line.split("=")[1].strip('"')
                        result.add_pass("发行版", distro)
                        break
        except Exception as e:
            result.add_warning("发行版检测", str(e))


def test_python_environment(result: TestResult):
    """测试Python环境"""
    print("\n[2] Python环境测试")
    
    required_version = (3, 10)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        result.add_pass("Python版本检查", f"{current_version[0]}.{current_version[1]}")
    else:
        result.add_fail("Python版本检查", f"需要 >={required_version[0]}.{required_version[1]}")
    
    try:
        import pip
        result.add_pass("pip", pip.__version__)
    except ImportError:
        result.add_fail("pip", "未安装")
    
    if IS_LINUX:
        venv_path = os.environ.get("VIRTUAL_ENV")
        if venv_path:
            result.add_pass("虚拟环境", venv_path)
        else:
            result.add_warning("虚拟环境", "未检测到激活的虚拟环境")


def test_core_dependencies(result: TestResult):
    """测试核心依赖"""
    print("\n[3] 核心依赖测试")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    for module, name in dependencies:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "未知")
            result.add_pass(name, version)
        except ImportError as e:
            result.add_fail(name, str(e))


def test_gpu_support(result: TestResult):
    """测试GPU支持"""
    print("\n[4] GPU支持测试")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            result.add_pass("CUDA可用", "是")
            result.add_pass("CUDA版本", torch.version.cuda)
            result.add_pass("cuDNN版本", torch.backends.cudnn.version())
            result.add_pass("GPU数量", str(torch.cuda.device_count()))
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                result.add_pass(f"GPU {i}", f"{name} ({memory:.1f} GB)")
            
            torch.cuda.empty_cache()
            result.add_pass("GPU缓存清理", "成功")
        else:
            result.add_warning("CUDA可用", "否 (将使用CPU模式)")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result.add_pass("MPS可用", "是 (Apple Silicon)")
        
    except Exception as e:
        result.add_fail("GPU测试", str(e))


def test_project_structure(result: TestResult):
    """测试项目结构"""
    print("\n[5] 项目结构测试")
    
    project_root = Path(__file__).resolve().parent.parent
    
    required_dirs = [
        "src",
        "scripts",
        "configs",
        "data",
        "results",
        "checkpoints",
        "logs",
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            result.add_pass(f"目录 {dir_name}/", "存在")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                result.add_pass(f"目录 {dir_name}/", "已创建")
            except Exception as e:
                result.add_fail(f"目录 {dir_name}/", str(e))
    
    required_files = [
        "requirements.txt",
        "src/__init__.py",
        "src/core/trainer.py",
        "src/methods/__init__.py",
        "scripts/run_experiment.py",
    ]
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            result.add_pass(f"文件 {file_name}", "存在")
        else:
            result.add_fail(f"文件 {file_name}", "不存在")


def test_module_imports(result: TestResult):
    """测试模块导入"""
    print("\n[6] 模块导入测试")
    
    modules = [
        ("src.core.trainer", "FederatedTrainer"),
        ("src.methods.fedavg", "FedAvgAggregator"),
        ("src.methods.dafa", "DAFAAggregator"),
        ("src.methods.dir_weight", "DirWeightAggregator"),
        ("src.data.cifar10", "get_cifar10_loaders"),
        ("src.models.resnet", "ResNet18"),
        ("src.utils.logger", "get_logger"),
        ("src.utils.checkpoint", "CheckpointManager"),
    ]
    
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
    for module_name, attr_name in modules:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            getattr(module, attr_name)
            result.add_pass(f"{module_name}.{attr_name}", "导入成功")
        except ImportError as e:
            result.add_fail(f"{module_name}.{attr_name}", f"导入失败: {e}")
        except AttributeError as e:
            result.add_fail(f"{module_name}.{attr_name}", f"属性不存在: {e}")


def test_file_permissions(result: TestResult):
    """测试文件权限"""
    print("\n[7] 文件权限测试")
    
    if not IS_LINUX:
        result.add_warning("文件权限测试", "仅在Linux系统上运行")
        return
    
    project_root = Path(__file__).resolve().parent.parent
    
    scripts = [
        "scripts/run_experiment.py",
        "scripts/run_five_stages.py",
        "scripts/analyze_results.py",
    ]
    
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            if os.access(script_path, os.X_OK):
                result.add_pass(f"执行权限 {script}", "有")
            else:
                try:
                    os.chmod(script_path, 0o755)
                    result.add_pass(f"执行权限 {script}", "已设置")
                except Exception as e:
                    result.add_warning(f"执行权限 {script}", str(e))
    
    shell_scripts = [
        "scripts/setup_env.sh",
        "scripts/run_experiment.sh",
        "scripts/run_quick.sh",
    ]
    
    for script in shell_scripts:
        script_path = project_root / script
        if script_path.exists():
            if os.access(script_path, os.X_OK):
                result.add_pass(f"执行权限 {script}", "有")
            else:
                try:
                    os.chmod(script_path, 0o755)
                    result.add_pass(f"执行权限 {script}", "已设置")
                except Exception as e:
                    result.add_warning(f"执行权限 {script}", str(e))


def test_data_loading(result: TestResult):
    """测试数据加载"""
    print("\n[8] 数据加载测试")
    
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        dummy_data = torch.randn(100, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (100,))
        dataset = TensorDataset(dummy_data, dummy_labels)
        
        loader = DataLoader(dataset, batch_size=32, num_workers=0)
        batch = next(iter(loader))
        
        result.add_pass("数据加载器", f"批次形状: {batch[0].shape}")
        
        if IS_LINUX:
            try:
                loader_mp = DataLoader(dataset, batch_size=32, num_workers=2)
                batch_mp = next(iter(loader_mp))
                result.add_pass("多进程数据加载", f"工作进程: 2")
            except Exception as e:
                result.add_warning("多进程数据加载", str(e))
        
    except Exception as e:
        result.add_fail("数据加载测试", str(e))


def test_model_creation(result: TestResult):
    """测试模型创建"""
    print("\n[9] 模型创建测试")
    
    try:
        import torch
        import torch.nn as nn
        
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.models.resnet import ResNet18
        from src.models.cnn import SimpleCNN
        
        model = ResNet18(num_classes=10)
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model(dummy_input)
        
        result.add_pass("ResNet18", f"输出形状: {output.shape}")
        
        model2 = SimpleCNN(num_classes=10)
        output2 = model2(dummy_input)
        
        result.add_pass("SimpleCNN", f"输出形状: {output2.shape}")
        
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            dummy_cuda = dummy_input.cuda()
            output_cuda = model_cuda(dummy_cuda)
            result.add_pass("GPU模型推理", f"输出形状: {output_cuda.shape}")
        
    except Exception as e:
        result.add_fail("模型创建测试", str(e))


def test_aggregation_methods(result: TestResult):
    """测试聚合方法"""
    print("\n[10] 聚合方法测试")
    
    try:
        import torch
        
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.methods.fedavg import FedAvgAggregator, FedAvgConfig
        from src.methods.dafa import DAFAAggregator, DAFAConfig
        from src.methods.dir_weight import DirWeightAggregator, DirWeightConfig
        from src.methods.base import ClientUpdate
        
        config = {"device": "cpu", "use_data_size_weighting": True}
        
        fedavg = FedAvgAggregator(FedAvgConfig(**config))
        result.add_pass("FedAvg初始化", "成功")
        
        dafa_config = config.copy()
        dafa_config.update({"gamma": 1.0, "beta": 0.9, "mu": 0.01})
        dafa = DAFAAggregator(DAFAConfig(**dafa_config))
        result.add_pass("DAFA初始化", "成功")
        
        dir_config = config.copy()
        dir_config.update({"gamma": 1.0, "mu": 0.01})
        dir_weight = DirWeightAggregator(DirWeightConfig(**dir_config))
        result.add_pass("Dir-Weight初始化", "成功")
        
        dummy_updates = [
            ClientUpdate(client_id=0, update=torch.randn(100), num_samples=100, loss=0.5),
            ClientUpdate(client_id=1, update=torch.randn(100), num_samples=100, loss=0.5),
        ]
        
        result.add_pass("ClientUpdate创建", "成功")
        
    except Exception as e:
        result.add_fail("聚合方法测试", str(e))


def test_config_loading(result: TestResult):
    """测试配置加载"""
    print("\n[11] 配置加载测试")
    
    try:
        import yaml
        
        project_root = Path(__file__).resolve().parent.parent
        
        config_files = [
            "configs/methods/dafa.yaml",
            "configs/methods/fedavg.yaml",
            "configs/datasets/cifar10.yaml",
        ]
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                result.add_pass(f"配置文件 {config_file}", "加载成功")
            else:
                result.add_warning(f"配置文件 {config_file}", "不存在")
        
    except Exception as e:
        result.add_fail("配置加载测试", str(e))


def test_logging(result: TestResult):
    """测试日志功能"""
    print("\n[12] 日志功能测试")
    
    try:
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.utils.logger import get_logger, setup_logger
        
        logger = get_logger("test")
        logger.info("测试日志消息")
        
        result.add_pass("日志器创建", "成功")
        
        log_file = project_root / "logs" / "test.log"
        setup_logger(log_file=str(log_file))
        
        logger2 = get_logger("test_file")
        logger2.info("测试文件日志")
        
        if log_file.exists():
            result.add_pass("文件日志", f"写入成功: {log_file}")
        else:
            result.add_warning("文件日志", "文件未创建")
        
    except Exception as e:
        result.add_fail("日志功能测试", str(e))


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("DAFA联邦学习框架 - Ubuntu环境测试")
    print("=" * 60)
    
    result = TestResult()
    
    test_system_info(result)
    test_python_environment(result)
    test_core_dependencies(result)
    test_gpu_support(result)
    test_project_structure(result)
    test_module_imports(result)
    test_file_permissions(result)
    test_data_loading(result)
    test_model_creation(result)
    test_aggregation_methods(result)
    test_config_loading(result)
    test_logging(result)
    
    return result.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
