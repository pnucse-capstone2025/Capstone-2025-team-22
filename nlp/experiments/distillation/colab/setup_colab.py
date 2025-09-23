import subprocess
import sys
import os

def install_requirements():
    """필요한 패키지들 설치"""
    print("📦 필요한 패키지들 설치 중...")
    
    packages = [
        'torch',
        'transformers',
        'TorchCRF',
        'scikit-learn',
        'matplotlib',
        'numpy',
        'sentencepiece',  # KoBERT 토크나이저용
        'tokenizers'      # 토크나이저 지원
    ]
    
    for package in packages:
        print(f"📥 {package} 설치 중...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 설치 실패: {e}")

def check_gpu():
    """GPU 사용 가능 여부 확인"""
    print("🔍 GPU 환경 확인 중...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능")
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("⚠️ CUDA 사용 불가 - CPU 모드로 실행됩니다")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다")
        return False

def setup_directories():
    """필요한 디렉토리 생성"""
    print("📁 디렉토리 구조 생성 중...")
    
    dirs = [
        '../results',
        '../results/models',
        '../results/plots',
        '../utils'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ {dir_path} 생성 완료")

def main():
    """메인 설정 함수"""
    print("🎯 Google Colab 환경 설정 시작!")
    print("=" * 50)
    
    # 1. 패키지 설치
    install_requirements()
    
    # 2. GPU 확인
    gpu_available = check_gpu()
    
    # 3. 디렉토리 생성
    setup_directories()
    
    print("\n" + "=" * 50)
    print("🎉 Colab 환경 설정 완료!")
    
    if gpu_available:
        print("🚀 GPU 가속 실험이 가능합니다!")
        print("💡 다음 명령으로 실험을 시작하세요:")
        print("   python real_distillation_gpu.py")
    else:
        print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
