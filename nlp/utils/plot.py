import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import argparse

def create_confusion_matrices(tp_model, fp_model, fn_model):
    """
    키워드 추출 모델의 혼동 행렬을 생성합니다.
    
    Args:
        tp_model: 모델의 True Positive 값
        fp_model: 모델의 False Positive 값
        fn_model: 모델의 False Negative 값
        
    Returns:
        numpy.ndarray: 혼동 행렬
    """
# Define confusion matrices
    cm_model = np.array([[tp_model, fp_model],
                        [fn_model, 0]])

    return cm_model 

def plot_confusion_matrix(cm, title, cmap='Blues'):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        cm: 혼동 행렬 (numpy array)
        title: 그래프 제목
        cmap: 색상 맵
    
    Returns:
        matplotlib.figure.Figure: 그래프 객체
    """
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title(title)
    plt.tight_layout()
    return fig

def calculate_metrics(tp, fp, fn):
    """
    성능 지표를 계산합니다.
    
    Args:
        tp: True Positive 값
        fp: False Positive 값
        fn: False Negative 값
        
    Returns:
        dict: 정밀도, 재현율, F1 점수를 포함하는 딕셔너리
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def print_metrics(metrics, model_name):
    """
    성능 지표를 출력합니다.
    
    Args:
        metrics: 성능 지표 딕셔너리
        model_name: 모델 이름
    """
    print(f"===== {model_name} 성능 지표 =====")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print()

def draw_confusion_matrix(tp_model, fp_model, fn_model, model_name):
    """
    혼동 행렬을 생성하고 시각화합니다.
    
    Args:
        tp_model: 모델의 True Positive 값
        fp_model: 모델의 False Positive 값
        fn_model: 모델의 False Negative 값
        model_name: 모델 이름
        
    Returns:
        matplotlib.figure.Figure: 그래프 객체
    """
    # Confusion matrix values
    # Format: [ [TP, FP], [FN, TN] ]
    # Since TN is not used in multilabel settings like keyword extraction, we omit it here.
    
    # 혼동 행렬 생성
    cm_model = create_confusion_matrices(
        tp_model, fp_model, fn_model
    )
    
    # 성능 지표 계산 및 출력
    metrics_model = calculate_metrics(tp_model, fp_model, fn_model)
    
    print_metrics(metrics_model, model_name)
    
    # 개별 혼동 행렬 시각화
    fig_model = plot_confusion_matrix(cm_model, f"{model_name} Confusion Matrix", "Blues")
    
    return fig_model

def parse_confusion_matrix(log_path):
    """
    로그 파일에서 혼동 행렬 데이터(TP, FP, FN)를 파싱합니다.
    
    Args:
        log_path: 로그 파일 경로
        
    Returns:
        tuple: (tp, fp, fn) 값들
    """
    tp_model = 0
    fp_model = 0
    fn_model = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # "Confusion Matrix - TP: X, FP: Y, FN: Z" 형식 처리
            if "Confusion Matrix" in line and "TP:" in line and "FP:" in line and "FN:" in line:
                try:
                    # "TP: X, FP: Y, FN: Z" 부분 추출
                    matrix_part = line.split("Confusion Matrix - ")[1].strip()
                    
                    # 개별 값 추출
                    tp_part = matrix_part.split("TP:")[1].split(",")[0].strip()
                    fp_part = matrix_part.split("FP:")[1].split(",")[0].strip()
                    
                    # FN은 마지막일 수 있으므로 comma 여부 확인
                    if "," in matrix_part.split("FN:")[1]:
                        fn_part = matrix_part.split("FN:")[1].split(",")[0].strip()
                    else:
                        fn_part = matrix_part.split("FN:")[1].strip()
                    
                    tp_model = int(tp_part)
                    fp_model = int(fp_part)
                    fn_model = int(fn_part)
                    # 성공적으로 파싱했으면 루프 종료
                    print(f"'Confusion Matrix' 포맷에서 파싱됨 - TP: {tp_model}, FP: {fp_model}, FN: {fn_model}")
                    break
                except Exception as e:
                    print(f"'Confusion Matrix' 포맷 파싱 오류 (계속 검색): {e}")
                    continue
            
            # 개별 키워드로 파싱 (이전 방식 유지)
            elif "TP:" in line and not "FP:" in line:
                try:
                    tp_model = int(line.split("TP:")[1].strip())
                except Exception:
                    pass
            elif "FP:" in line and not "TP:" in line:
                try:
                    fp_model = int(line.split("FP:")[1].strip())
                except Exception:
                    pass
            elif "FN:" in line and not "FP:" in line:
                try:
                    fn_model = int(line.split("FN:")[1].strip())
                except Exception:
                    pass
    
    print(f"파싱된 혼동 행렬 값 - TP: {tp_model}, FP: {fp_model}, FN: {fn_model}")
    return tp_model, fp_model, fn_model

def parse_log(log_path):
    """
    로그 파일을 파싱하여 학습 단계, 손실, 정확도 데이터를 추출합니다.
    
    Args:
        log_path: 로그 파일 경로
        
    Returns:
        tuple: (train_steps, train_losses, train_accs, val_steps, val_losses, val_accs)
    """
    train_steps, train_losses, train_accs = [], [], []
    val_steps, val_losses, val_accs = [], [], []

    # 학습 데이터 패턴 (손실 및 정확도)
    train_loss_acc_pattern = re.compile(r"Step: (\d+), Loss: ([0-9\.]+), Acc: ([0-9\.]+)")
    # 학습 데이터 패턴 (손실만)
    train_loss_pattern = re.compile(r"Step: (\d+), Loss: ([0-9\.]+)")
    # 검증 데이터 패턴
    val_pattern = re.compile(r"Total Step: \d+, Total Val loss: ([0-9\.]+), Acc: ([0-9\.]+)")

    with open(log_path, 'r') as f:
        i = 1
        for line in f:
            # 학습 손실 및 정확도 추출
            m_train_loss_acc = train_loss_acc_pattern.search(line)
            if m_train_loss_acc:
                step = int(m_train_loss_acc.group(1))
                loss = float(m_train_loss_acc.group(2))
                acc = float(m_train_loss_acc.group(3))
                
                train_steps.append(step)
                train_losses.append(loss)
                train_accs.append(acc * 100)  # 백분율로 변환
                continue
            
            # 학습 손실만 추출 (정확도 없는 경우)
            m_train_loss = train_loss_pattern.search(line)
            if m_train_loss and not m_train_loss_acc:
                step = int(m_train_loss.group(1))
                loss = float(m_train_loss.group(2))
                
                train_steps.append(step)
                train_losses.append(loss)
                continue
                
            # 검증 데이터 추출
            m_val = val_pattern.search(line)
            if m_val:
                # 에포크마다 검증 수행 가정, 에포크당 300 스텝으로 가정
                val_steps.append(i * 300)
                val_losses.append(float(m_val.group(1)))
                val_accs.append(float(m_val.group(2)) * 100)  # 백분율로 변환
                i += 1

    return train_steps, train_losses, train_accs, val_steps, val_losses, val_accs

def plot_accuracy(train_steps, train_accs, val_steps, val_accs, save_path=None, show=True):
    """
    학습 및 검증 정확도 그래프를 그립니다.
    
    Args:
        train_steps: 학습 단계 리스트
        train_accs: 학습 정확도 리스트
        val_steps: 검증 단계 리스트
        val_accs: 검증 정확도 리스트
        save_path: 그래프 저장 경로 (None이면 저장하지 않음)
        show: 그래프 표시 여부
        
    Returns:
        matplotlib.figure.Figure: 그래프 객체
    """
    fig = plt.figure(figsize=(12, 6))
    
    # 학습 정확도 그래프 (모든 단계)
    if train_steps and train_accs:
        plt.plot(train_steps, train_accs, color='blue', label='Training Accuracy (per step)', alpha=0.7, zorder=0)
    
    # 검증 정확도 그래프 (에포크별 평균)
    if val_steps and val_accs and len(val_steps) > 0:
        # 점 연결 선 그리기 (점보다 뒤에)
        plt.plot(val_steps, val_accs, color='red', linestyle='-', zorder=1)
        # 점 그리기 (선보다 앞에)
        plt.scatter(val_steps, val_accs, color='red', s=100, label='Validation Accuracy (per epoch)', zorder=2)
    
    # 각 검증 포인트에 값 표시
    if val_steps and val_accs and len(val_steps) > 0:
        for i, (x, y) in enumerate(zip(val_steps, val_accs)):
            plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
    # 최대 검증 정확도 포인트에 특별 표시
    if val_accs and len(val_accs) > 0:
        max_index = val_accs.index(max(val_accs))
        best_x, best_y = val_steps[max_index], val_accs[max_index]
        plt.annotate('Best Model!', 
                    xy=(best_x, best_y),
                    xytext=(20, -40),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path)
        print(f"정확도 그래프 저장 완료: {save_path}")
    
    if show:
        plt.show()
    
    return fig

def plot_loss(train_steps, train_losses, val_steps, val_losses, save_path=None, show=True):
    """
    학습 및 검증 손실 그래프를 그립니다.
    
    Args:
        train_steps: 학습 단계 리스트
        train_losses: 학습 손실 리스트
        val_steps: 검증 단계 리스트
        val_losses: 검증 손실 리스트
        save_path: 그래프 저장 경로 (None이면 저장하지 않음)
        show: 그래프 표시 여부
        
    Returns:
        matplotlib.figure.Figure: 그래프 객체
    """
    fig = plt.figure(figsize=(12, 6))
    
    # 검증 손실 그래프 (에포크별 평균)
    if val_steps and val_losses and len(val_steps) > 0:
        # 점 연결 선 그리기 (점보다 뒤에)
        plt.plot(val_steps, val_losses, color='red', linestyle='-', zorder=1)
        # 점 그리기 (선보다 앞에)
        plt.scatter(val_steps, val_losses, color='red', s=100, label='Validation Loss (avg per epoch)', zorder=2)
    
    # 학습 손실 그래프 (모든 단계) - 검증보다 뒤에
    if train_steps and train_losses:
        plt.plot(train_steps, train_losses, color='blue', label='Training Loss (per step)', alpha=0.7, zorder=0)
    
    # 각 검증 포인트에 값 표시
    if val_steps and val_losses and len(val_steps) > 1:
        for i, (x, y) in enumerate(zip(val_steps, val_losses)):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
    # 최소 검증 손실 포인트에 특별 표시
    if val_losses and len(val_losses) > 0:
        min_index = val_losses.index(min(val_losses))
        best_x, best_y = val_steps[min_index], val_losses[min_index]
        plt.annotate('Best Model!', 
                    xy=(best_x, best_y),
                    xytext=(20, 40),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path)
        print(f"손실 그래프 저장 완료: {save_path}")
    
    if show:
        plt.show()
    
    return fig

def plot_training_results(log_path, output_dir=None, show=True):
    """
    로그 파일에서 학습 결과를 추출하여 손실 및 정확도 그래프를 그립니다.
    
    Args:
        log_path: 로그 파일 경로
        output_dir: 그래프 저장 디렉토리 (None이면 현재 디렉토리)
        show: 그래프 표시 여부
        
    Returns:
        tuple: (loss_figure, accuracy_figure)
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    # 로그 파일 파싱
    train_steps, train_losses, train_accs, val_steps, val_losses, val_accs = parse_log(log_path)
    
    # 손실 그래프
    loss_fig = None
    if train_losses or val_losses:
        loss_save_path = os.path.join(output_dir, 'training_vs_validation_loss.png')
        loss_fig = plot_loss(train_steps, train_losses, val_steps, val_losses, loss_save_path, show)
    
    # 정확도 그래프
    acc_fig = None
    if train_accs or val_accs:
        acc_save_path = os.path.join(output_dir, 'training_vs_validation_accuracy.png')
        acc_fig = plot_accuracy(train_steps, train_accs, val_steps, val_accs, acc_save_path, show)
    
    return loss_fig, acc_fig

def parse_metrics(log_path):
    """
    로그 파일에서 정밀도(P), 재현율(R), F1 점수를 파싱합니다.
    
    지원하는 로그 형식:
    1. 2025-05-15 06:22:38,115 - keybert_test - DEBUG - Step: 12/38, P: 0.0990, R: 0.0754, F1: 0.0856
    2. 2025-05-15 06:19:42,228 - kokeybert_test - DEBUG - Step: 3/38, Loss: 10.5017, Acc: 0.8397, P: 0.5780, R: 0.5860, F1: 0.5820
    
    Args:
        log_path: 로그 파일 경로
        
    Returns:
        tuple: (steps, precision, recall, f1) - 각각 리스트 형태
    """
    steps = []
    precision = []
    recall = []
    f1 = []
    
    # 두 가지 패턴을 처리하는 정규표현식
    # Loss & Acc 포함 패턴: Step: X/Y, Loss: Z, Acc: W, P: A, R: B, F1: C
    metrics_pattern2 = re.compile(r"Step: (\d+)/\d+, Loss: [0-9\.]+, Acc: [0-9\.]+, P: ([0-9\.]+), R: ([0-9\.]+), F1: ([0-9\.]+)")
    
    # 기본 패턴: Step: X/Y, P: A, R: B, F1: C
    metrics_pattern1 = re.compile(r"Step: (\d+)/\d+, P: ([0-9\.]+), R: ([0-9\.]+), F1: ([0-9\.]+)")
    
    with open(log_path, 'r') as f:
        for line in f:
            # 패턴 2 먼저 확인 (더 구체적인 패턴)
            m2 = metrics_pattern2.search(line)
            if m2:
                step = int(m2.group(1))
                p_value = float(m2.group(2))
                r_value = float(m2.group(3))
                f1_value = float(m2.group(4))
                
                steps.append(step)
                precision.append(p_value)
                recall.append(r_value)
                f1.append(f1_value)
                continue
            
            # 패턴 1 확인
            m1 = metrics_pattern1.search(line)
            if m1:
                step = int(m1.group(1))
                p_value = float(m1.group(2))
                r_value = float(m1.group(3))
                f1_value = float(m1.group(4))
                
                steps.append(step)
                precision.append(p_value)
                recall.append(r_value)
                f1.append(f1_value)
    
    # 결과 출력
    if steps:
        print(f"파싱된 성능 지표: {len(steps)}개 (Step {min(steps)}~{max(steps)})")
    else:
        print("파싱된 성능 지표가 없습니다.")
    
    return steps, precision, recall, f1

def plot_metrics(data_list, labels=None, colors=None, title='모델 성능 지표', 
                save_path=None, show=True, figsize=(18, 6)):
    """
    정밀도(P), 재현율(R), F1 점수를 시각화합니다.
    
    Args:
        data_list: 데이터 리스트. 각 항목은 (steps, precision, recall, f1) 튜플.
                  예: [(steps1, p1, r1, f1), (steps2, p2, r2, f2), ...]
        labels: 각 데이터에 대한 레이블 리스트 (모델 이름 등)
        colors: 각 데이터에 대한 색상 리스트
        title: 그래프 전체 제목
        save_path: 그래프 저장 경로 (None이면 저장하지 않음)
        show: 그래프 표시 여부
        figsize: 그래프 크기
        
    Returns:
        matplotlib.figure.Figure: 그래프 객체
    """
    if not data_list:
        print("시각화할 데이터가 없습니다.")
        return None
    
    # 기본값 설정
    if labels is None:
        labels = [f"model {i+1}" for i in range(len(data_list))]
    
    if colors is None:
        # 기본 색상표 (최대 10개)
        default_colors = ['blue', 'orange', 'green', 'red', 'purple', 
                         'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = [default_colors[i % len(default_colors)] for i in range(len(data_list))]
    
    # 그래프 생성 (수평 정렬)
    fig = plt.figure(figsize=figsize)
    # Precision (첫 번째 열)
    plt.subplot(1, 3, 1)
    for i, (steps, precision, _, _) in enumerate(data_list):
        plt.plot(steps, precision, label=labels[i], color=colors[i])
    plt.title('Precision')
    plt.xlabel('Step')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Recall (두 번째 열)
    plt.subplot(1, 3, 2)
    for i, (steps, _, recall, _) in enumerate(data_list):
        plt.plot(steps, recall, label=labels[i], color=colors[i])
    plt.title('Recall')
    plt.xlabel('Step')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # F1 Score (세 번째 열)
    plt.subplot(1, 3, 3)
    for i, (steps, _, _, f1) in enumerate(data_list):
        plt.plot(steps, f1, label=labels[i], color=colors[i])
    plt.title('F1 Score')
    plt.xlabel('Step')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path)
        print(f"model performance metrics graph saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='학습 그래프 및 혼동 행렬 시각화')
    parser.add_argument('--logfile', help='로그 파일 경로')
    parser.add_argument('--output_dir', default=None, help='출력 디렉토리 경로')
    parser.add_argument('--model_name', default='Model', help='모델의 이름')
    parser.add_argument('--parse_type', default='loss_acc', 
                        help='파싱 타입 (loss_acc, confusion_matrix, metrics)')
    parser.add_argument('--compare', action='store_true', 
                        help='여러 로그 파일 비교 (--logfile은 콤마로 구분된 파일 목록)')
    parser.add_argument('--labels', help='비교할 모델 이름들 (콤마로 구분)')
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 파싱 타입에 따라 처리
    if args.parse_type == 'loss_acc':
        # 학습/검증 손실 및 정확도 그래프 그리기
        plot_training_results(args.logfile, args.output_dir)
    
    elif args.parse_type == 'confusion_matrix':
        # 로그 파일에서 혼동 행렬 데이터 파싱
        tp_model, fp_model, fn_model = parse_confusion_matrix(args.logfile)
        
        # 혼동 행렬 그래프 생성
        fig = draw_confusion_matrix(tp_model, fp_model, fn_model, args.model_name)
        
        # 그래프 저장
        if args.output_dir:
            save_path = os.path.join(args.output_dir, f'{args.model_name.lower()}_confusion_matrix.png')
            fig.savefig(save_path)
            print(f"혼동 행렬 저장 완료: {save_path}")
        
        plt.show()
        
    elif args.parse_type == 'metrics':
        if args.compare:
            # 여러 로그 파일 비교
            log_files = args.logfile.split(',')
            labels = args.labels.split(',') if args.labels else [f"model {i+1}" for i in range(len(log_files))]
            
            data_list = []
            for log_file in log_files:
                data = parse_metrics(log_file.strip())
                if data[0]:  # steps가 있는 경우만 추가
                    data_list.append(data)
            
            if data_list:
                save_path = os.path.join(args.output_dir, 'model_performance_comparison.png') if args.output_dir else None
                fig = plot_metrics(data_list, labels=labels, title='Model Performance Comparison', 
                                 save_path=save_path)
        else:
            # 단일 로그 파일
            steps, precision, recall, f1 = parse_metrics(args.logfile)
            if steps:
                data_list = [(steps, precision, recall, f1)]
                save_path = os.path.join(args.output_dir, f'{args.model_name.lower()}_metrics.png') if args.output_dir else None
                fig = plot_metrics(data_list, labels=[args.model_name], 
                                 title=f'{args.model_name} model performance metrics', 
                                 save_path=save_path)
