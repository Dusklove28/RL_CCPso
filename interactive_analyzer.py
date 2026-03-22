"""
交互式训练结果分析工具（无需 TensorFlow）
使用 h5py 直接读取和分析 .h5 文件
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


def list_all_tasks(task_dir='data/task'):
    """列出所有训练任务"""
    task_path = Path(task_dir)
    tasks = []
    
    print("\n" + "="*60)
    print("可用的训练任务")
    print("="*60)
    
    for item in task_path.iterdir():
        if item.is_dir() and len(item.name) == 32:
            actor_files = list(item.glob('ddpg_actor_episode*.h5'))
            
            if actor_files:
                episodes = []
                for f in actor_files:
                    try:
                        start = f.name.index('episode') + 7
                        end = f.name.index('_', start)
                        episode = int(f.name[start:end])
                        episodes.append(episode)
                    except:
                        pass
                
                if episodes:
                    tasks.append({
                        'hash': item.name,
                        'episodes': sorted(episodes),
                        'max_episode': max(episodes),
                        'file_count': len(actor_files)
                    })
    
    # 按最大轮次排序
    tasks.sort(key=lambda x: x['max_episode'], reverse=True)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. {task['hash']}")
        print(f"   最大轮次：{task['max_episode']}")
        print(f"   检查点数：{task['file_count']}")
        print(f"   轮次范围：{min(task['episodes'])} - {max(task['episodes'])}")
    
    return tasks


def read_h5_weights(h5_file_path):
    """读取 H5 文件的权重信息"""
    weights_info = []
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            for layer_name in f.keys():
                layer_group = f[layer_name]
                layer_weights = {}
                
                for weight_name in layer_group.keys():
                    weight_data = layer_group[weight_name][()]
                    if isinstance(weight_data, np.ndarray):
                        layer_weights[weight_name] = {
                            'shape': weight_data.shape,
                            'size': weight_data.size,
                            'mean': float(np.mean(weight_data)),
                            'std': float(np.std(weight_data)),
                            'min': float(np.min(weight_data)),
                            'max': float(np.max(weight_data)),
                            'sparsity': float(np.mean(np.abs(weight_data) < 1e-6))
                        }
                
                if layer_weights:
                    weights_info.append({
                        'layer_name': layer_name,
                        'weights': layer_weights
                    })
        
        return weights_info
        
    except Exception as e:
        print(f"读取失败：{e}")
        return None


def analyze_single_model(model_path):
    """分析单个模型文件"""
    print(f"\n分析模型：{model_path.name}")
    print(f"{'-'*60}")
    
    weights_info = read_h5_weights(model_path)
    if not weights_info:
        print("无法读取模型文件")
        return
    
    # 统计信息
    total_params = 0
    print(f"\n模型结构:")
    print(f"{'层名':<30} {'权重名':<15} {'形状':<20} {'参数量':>10}")
    print(f"{'-'*80}")
    
    for layer_info in weights_info:
        layer_name = layer_info['layer_name']
        for weight_name, weight_data in layer_info['weights'].items():
            print(f"{layer_name:<30} {weight_name:<15} "
                  f"{str(weight_data['shape']):<20} {weight_data['size']:>10,}")
            total_params += weight_data['size']
    
    print(f"{'-'*80}")
    print(f"总参数量：{total_params:,}")
    
    # 权重分布统计
    print(f"\n权重分布统计:")
    all_means = []
    all_stds = []
    all_sparsity = []
    
    for layer_info in weights_info:
        for weight_name, weight_data in layer_info['weights'].items():
            all_means.append(weight_data['mean'])
            all_stds.append(weight_data['std'])
            all_sparsity.append(weight_data['sparsity'])
    
    if all_means:
        print(f"  平均权重均值：{np.mean(all_means):.6f}")
        print(f"  平均权重标准差：{np.mean(all_stds):.6f}")
        print(f"  平均稀疏度：{np.mean(all_sparsity):.4f} ({np.mean(all_sparsity)*100:.2f}%)")
    
    return weights_info


def compare_models(task_hash, episodes, task_dir='data/task'):
    """比较多个训练轮次的模型"""
    task_path = Path(task_dir) / task_hash
    
    print(f"\n{'='*60}")
    print(f"比较任务 {task_hash[:16]}... 的不同训练轮次")
    print(f"{'='*60}")
    
    results = []
    
    for episode in episodes:
        model_file = task_path / f'ddpg_actor_episode{episode}_round0.h5'
        if model_file.exists():
            weights_info = read_h5_weights(model_file)
            if weights_info:
                total_params = sum(
                    w_data['size'] 
                    for layer_info in weights_info 
                    for w_data in layer_info['weights'].values()
                )
                
                results.append({
                    'episode': episode,
                    'file': model_file.name,
                    'layers': len(weights_info),
                    'params': total_params
                })
    
    # 打印对比表
    print(f"\n{'轮次':<10} {'层数':<10} {'参数量':<15} {'文件名':<40}")
    print(f"{'-'*80}")
    for result in results:
        print(f"{result['episode']:<10} {result['layers']:<10} "
              f"{result['params']:<15,} {result['file']:<40}")
    
    return results


def plot_training_progress(task_hash, task_dir='data/task', save_dir='analysis_output'):
    """绘制训练进度图"""
    task_path = Path(task_dir) / task_hash
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 收集所有 actor 文件
    actor_files = list(task_path.glob('ddpg_actor_episode*.h5'))
    
    episodes = []
    params_list = []
    
    for model_file in sorted(actor_files):
        try:
            start = model_file.name.index('episode') + 7
            end = model_file.name.index('_', start)
            episode = int(model_file.name[start:end])
            
            weights_info = read_h5_weights(model_file)
            if weights_info:
                total_params = sum(
                    w_data['size'] 
                    for layer_info in weights_info 
                    for w_data in layer_info['weights'].values()
                )
                
                episodes.append(episode)
                params_list.append(total_params)
        except:
            pass
    
    if not episodes:
        print("没有找到训练数据")
        return
    
    # 绘制图表
    plt.figure(figsize=(12, 5))
    
    # 子图 1: 参数量变化
    plt.subplot(1, 2, 1)
    plt.plot(episodes, params_list, 'o-', linewidth=2, markersize=6)
    plt.xlabel('训练轮次')
    plt.ylabel('参数量')
    plt.title(f'训练进度 - 参数量变化\n任务：{task_hash[:16]}...')
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 训练轮次分布
    plt.subplot(1, 2, 2)
    plt.hist(episodes, bins=min(10, len(episodes)), edgecolor='black')
    plt.xlabel('训练轮次')
    plt.ylabel('检查点数量')
    plt.title('训练检查点分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{task_hash[:16]}_training_progress.png', dpi=300)
    plt.close()
    
    print(f"\n✓ 训练进度图已保存到：{output_dir / f'{task_hash[:16]}_training_progress.png'}")


def interactive_menu():
    """交互式菜单"""
    print("\n" + "="*60)
    print("DDPG 训练结果分析工具（交互式）")
    print("="*60)
    
    while True:
        print("\n请选择操作:")
        print("1. 查看所有训练任务")
        print("2. 分析单个模型文件")
        print("3. 比较同一任务的不同训练轮次")
        print("4. 绘制训练进度图")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == '1':
            tasks = list_all_tasks()
            
        elif choice == '2':
            task_hash = input("请输入任务哈希 (前 16 位): ").strip()
            episode = input("请输入训练轮次: ").strip()
            
            model_path = Path(f'data/task/{task_hash}/ddpg_actor_episode{episode}_round0.h5')
            if model_path.exists():
                analyze_single_model(model_path)
            else:
                print("模型文件不存在")
        
        elif choice == '3':
            task_hash = input("请输入任务哈希 (前 32 位): ").strip()
            episodes_str = input("请输入要比较的轮次 (用逗号分隔，如 20,40,60,120): ").strip()
            episodes = [int(e.strip()) for e in episodes_str.split(',')]
            
            compare_models(task_hash, episodes)
        
        elif choice == '4':
            task_hash = input("请输入任务哈希 (前 32 位): ").strip()
            plot_training_progress(task_hash)
        
        elif choice == '5':
            print("\n再见！")
            break
        
        else:
            print("无效的选项，请重新选择")


def quick_analyze_126_episodes():
    """快速分析 126 轮以内的所有任务"""
    print("="*60)
    print("快速分析 126 轮以内的训练任务")
    print("="*60)
    
    tasks = list_all_tasks()
    
    # 筛选有 126 轮以内数据的任务
    relevant_tasks = [t for t in tasks if t['max_episode'] >= 20]
    
    print(f"\n找到 {len(relevant_tasks)} 个相关任务")
    
    for task in relevant_tasks[:5]:  # 只显示前 5 个
        task_hash = task['hash']
        print(f"\n{'='*60}")
        print(f"任务：{task_hash}")
        print(f"最大轮次：{task['max_episode']}")
        print(f"{'='*60}")
        
        # 分析前 126 轮
        episodes_to_analyze = [e for e in task['episodes'] if e <= 126]
        
        if episodes_to_analyze:
            # 选择几个关键轮次
            key_episodes = []
            for threshold in [20, 40, 60, 80, 100, 120]:
                for e in episodes_to_analyze:
                    if e <= threshold and e not in key_episodes:
                        key_episodes.append(e)
                        break
            
            compare_models(task_hash, key_episodes)
            plot_training_progress(task_hash)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == '__main__':
    # 可以选择交互式模式或快速分析模式
    # interactive_menu()  # 交互式模式
    quick_analyze_126_episodes()  # 快速分析模式
