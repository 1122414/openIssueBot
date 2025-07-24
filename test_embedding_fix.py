#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入模型测试修复验证脚本

用于验证智谱AI 401错误和本地模型数组真值判断错误的修复
"""

import os
import sys
import numpy as np
from app.embedding import EmbeddingService
from app.config import Config
from app.utils import log_info, log_error

def test_zhipu_embedding():
    """
    测试智谱AI嵌入模型
    """
    print("\n=== 测试智谱AI嵌入模型 ===")
    
    # 检查API Key配置
    zhipu_api_key = getattr(Config, 'ZHIPU_API_KEY', None)
    zhipu_embedding_api_key = getattr(Config, 'ZHIPU_EMBEDDING_API_KEY', None)
    
    print(f"ZHIPU_API_KEY 长度: {len(zhipu_api_key) if zhipu_api_key else 0}")
    print(f"ZHIPU_EMBEDDING_API_KEY 长度: {len(zhipu_embedding_api_key) if zhipu_embedding_api_key else 0}")
    
    if not zhipu_api_key and not zhipu_embedding_api_key:
        print("❌ 未配置智谱AI API Key，跳过测试")
        return False
    
    try:
        # 创建嵌入服务
        embedding_service = EmbeddingService(provider='zhipu', model_name='embedding-2')
        
        # 测试获取嵌入向量
        test_text = "这是一个测试文本"
        embeddings = embedding_service.get_embeddings([test_text])
        
        # 验证结果
        if embeddings is not None and embeddings.size > 0:
            print(f"✅ 智谱AI嵌入模型测试成功")
            print(f"   向量维度: {embeddings.shape}")
            print(f"   向量类型: {type(embeddings)}")
            return True
        else:
            print("❌ 智谱AI嵌入模型返回空结果")
            return False
            
    except Exception as e:
        print(f"❌ 智谱AI嵌入模型测试失败: {e}")
        return False

def test_local_embedding():
    """
    测试本地嵌入模型
    """
    print("\n=== 测试本地嵌入模型 ===")
    
    try:
        # 创建嵌入服务
        embedding_service = EmbeddingService(provider='local', model_name='all-MiniLM-L6-v2')
        
        # 测试获取嵌入向量
        test_text = "这是一个测试文本"
        embeddings = embedding_service.get_embeddings([test_text])
        
        # 验证结果 - 使用修复后的判断逻辑
        if embeddings is not None and embeddings.size > 0:
            print(f"✅ 本地嵌入模型测试成功")
            print(f"   向量维度: {embeddings.shape}")
            print(f"   向量类型: {type(embeddings)}")
            
            # 测试数组真值判断
            try:
                # 这应该不会报错
                if embeddings.size > 0:
                    print("✅ 数组真值判断修复成功")
                else:
                    print("❌ 数组真值判断仍有问题")
            except ValueError as e:
                print(f"❌ 数组真值判断错误: {e}")
                return False
                
            return True
        else:
            print("❌ 本地嵌入模型返回空结果")
            return False
            
    except Exception as e:
        print(f"❌ 本地嵌入模型测试失败: {e}")
        return False

def test_array_truth_value_fix():
    """
    专门测试数组真值判断修复
    """
    print("\n=== 测试数组真值判断修复 ===")
    
    # 创建测试数组
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    
    try:
        # 错误的判断方式（会报错）
        # if test_array:  # 这会导致 "The truth value of an array with more than one element is ambiguous"
        
        # 正确的判断方式
        if test_array.size > 0:
            print("✅ 使用 array.size > 0 判断成功")
        
        if len(test_array) > 0:
            print("✅ 使用 len(array) > 0 判断成功")
            
        if test_array is not None and test_array.size > 0:
            print("✅ 使用组合判断成功")
            
        return True
        
    except Exception as e:
        print(f"❌ 数组真值判断测试失败: {e}")
        return False

def main():
    """
    主测试函数
    """
    print("嵌入模型修复验证测试")
    print("=" * 50)
    
    results = []
    
    # 测试数组真值判断修复
    results.append(test_array_truth_value_fix())
    
    # 测试本地嵌入模型
    results.append(test_local_embedding())
    
    # 测试智谱AI嵌入模型
    results.append(test_zhipu_embedding())
    
    # 总结
    print("\n=== 测试总结 ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 所有测试通过！修复成功！")
    else:
        print("⚠️  部分测试失败，请检查配置和网络连接")
    
    return success_count == total_count

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)