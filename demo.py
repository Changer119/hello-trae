#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气助手演示脚本
展示如何在代码中使用WeatherAssistant类
"""

from weather_assistant import WeatherAssistant

def demo_weather_queries():
    """
    演示各种天气查询
    """
    # 请在这里填入您的API密钥
    API_KEY = "your_api_key_here"
    
    if API_KEY == "your_api_key_here":
        print("❌ 请先在demo.py中设置您的API密钥")
        print("获取API密钥: https://openweathermap.org/api")
        return
    
    # 创建天气助手实例
    assistant = WeatherAssistant(API_KEY)
    
    # 演示查询列表
    demo_queries = [
        "北京的天气怎么样？",
        "上海明天的天气",
        "广州未来3天的天气预报",
        "深圳后天天气如何",
        "杭州未来5天天气"
    ]
    
    print("🌤️  天气助手演示")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 查询 {i}: {query}")
        print("-" * 30)
        
        try:
            result = assistant.query_weather(query)
            print(result)
        except Exception as e:
            print(f"❌ 查询失败: {e}")
        
        print("=" * 50)

def demo_direct_api_calls():
    """
    演示直接调用API方法
    """
    # 请在这里填入您的API密钥
    API_KEY = "your_api_key_here"
    
    if API_KEY == "your_api_key_here":
        print("❌ 请先在demo.py中设置您的API密钥")
        return
    
    assistant = WeatherAssistant(API_KEY)
    
    print("\n🔧 直接API调用演示")
    print("=" * 50)
    
    # 获取北京坐标
    print("\n1. 获取城市坐标:")
    coords = assistant.get_coordinates("北京")
    if coords:
        lat, lon = coords
        print(f"北京坐标: 纬度 {lat}, 经度 {lon}")
        
        # 获取当前天气
        print("\n2. 获取当前天气:")
        current_weather = assistant.get_current_weather(lat, lon)
        if current_weather:
            formatted = assistant.format_current_weather(current_weather, "北京")
            print(formatted)
        
        # 获取天气预报
        print("\n3. 获取天气预报:")
        forecast = assistant.get_forecast(lat, lon, 3)
        if forecast:
            formatted = assistant.format_forecast(forecast, "北京", 3)
            print(formatted)
    else:
        print("❌ 无法获取城市坐标")

def demo_natural_language_parsing():
    """
    演示自然语言解析功能
    """
    assistant = WeatherAssistant("dummy_key")  # 只用于解析，不需要真实API密钥
    
    print("\n🧠 自然语言解析演示")
    print("=" * 50)
    
    test_queries = [
        "北京的天气怎么样？",
        "上海明天的天气",
        "广州未来3天的天气预报",
        "深圳后天天气如何",
        "杭州未来5天天气",
        "今天武汉天气",
        "成都明天会下雨吗"
    ]
    
    for query in test_queries:
        parsed = assistant.parse_natural_language(query)
        print(f"\n查询: {query}")
        print(f"解析结果: {parsed}")

if __name__ == "__main__":
    print("选择演示模式:")
    print("1. 完整天气查询演示")
    print("2. 直接API调用演示")
    print("3. 自然语言解析演示")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == "1":
        demo_weather_queries()
    elif choice == "2":
        demo_direct_api_calls()
    elif choice == "3":
        demo_natural_language_parsing()
    else:
        print("❌ 无效选择")