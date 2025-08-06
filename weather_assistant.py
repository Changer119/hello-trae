#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气助手 - 支持自然语言查询天气预报
使用OpenWeatherMap API获取天气数据
"""

import requests
import json
import re
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# 尝试导入配置文件
try:
    from config import API_KEY, DEFAULT_CITY, API_SETTINGS, QUERY_SETTINGS
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    API_KEY = None
    DEFAULT_CITY = "北京"
    API_SETTINGS = {
        "base_url": "http://api.openweathermap.org/data/2.5",
        "geocoding_url": "http://api.openweathermap.org/geo/1.0",
        "units": "metric",
        "lang": "zh_cn"
    }
    QUERY_SETTINGS = {
        "max_forecast_days": 5,
        "default_forecast_days": 3
    }

class WeatherAssistant:
    def __init__(self, api_key: str):
        """
        初始化天气助手
        
        Args:
            api_key: OpenWeatherMap API密钥
        """
        self.api_key = api_key
        self.base_url = API_SETTINGS["base_url"]
        self.geocoding_url = API_SETTINGS["geocoding_url"]
        self.units = API_SETTINGS["units"]
        self.lang = API_SETTINGS["lang"]
        
    def get_coordinates(self, city_name: str) -> Optional[Tuple[float, float]]:
        """
        根据城市名称获取经纬度坐标
        
        Args:
            city_name: 城市名称
            
        Returns:
            (纬度, 经度) 或 None
        """
        try:
            url = f"{self.geocoding_url}/direct"
            params = {
                'q': city_name,
                'limit': 1,
                'appid': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data:
                return data[0]['lat'], data[0]['lon']
            return None
            
        except Exception as e:
            print(f"获取坐标时出错: {e}")
            return None
    
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        获取当前天气信息
        
        Args:
            lat: 纬度
            lon: 经度
            
        Returns:
            天气数据字典或None
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': self.units,
                'lang': self.lang
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"获取天气信息时出错: {e}")
            return None
    
    def get_forecast(self, lat: float, lon: float, days: int = 5) -> Optional[Dict]:
        """
        获取天气预报
        
        Args:
            lat: 纬度
            lon: 经度
            days: 预报天数
            
        Returns:
            预报数据字典或None
        """
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': self.units,
                'lang': self.lang
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"获取天气预报时出错: {e}")
            return None
    
    def parse_natural_language(self, query: str) -> Dict[str, any]:
        """
        解析自然语言查询
        
        Args:
            query: 用户的自然语言查询
            
        Returns:
            解析结果字典
        """
        result = {
            'city': None,
            'query_type': 'current',  # current, forecast
            'days': 1
        }
        
        # 先移除时间词汇，再提取城市名称
        query_clean = query
        time_words = ['今天', '明天', '后天', '未来', '几天', '天', '的天气', '天气', '怎么样', '如何', '会下雨吗', '预报']
        
        # 提取城市名称的策略：
        # 1. 直接匹配常见城市名
        common_cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '西安', '重庆', 
                        '天津', '苏州', '青岛', '大连', '厦门', '宁波', '无锡', '长沙', '郑州', '济南']
        
        for city in common_cities:
            if city in query:
                result['city'] = city
                break
        
        # 2. 如果没有找到常见城市，使用正则表达式
        if not result['city']:
            city_patterns = [
                r'([\u4e00-\u9fa5]{2,}市?)的天气',  # 北京的天气
                r'([\u4e00-\u9fa5]{2,}市?)天气',   # 北京天气
                r'([\u4e00-\u9fa5]{2,}市?)(?:明天|后天)',  # 北京明天
                r'([A-Za-z\s]{2,})(?:的天气|天气)',  # 英文城市名
            ]
            
            for pattern in city_patterns:
                match = re.search(pattern, query)
                if match:
                    city_name = match.group(1).strip()
                    # 过滤掉时间词汇和无效词汇
                    if city_name not in time_words and len(city_name) >= 2:
                        result['city'] = city_name
                        break
        
        # 判断查询类型
        if any(word in query for word in ['明天', '后天', '未来', '预报', '几天']):
            result['query_type'] = 'forecast'
            
            # 提取天数
            day_match = re.search(r'(\d+)天', query)
            if day_match:
                result['days'] = min(int(day_match.group(1)), QUERY_SETTINGS["max_forecast_days"])
            elif '明天' in query:
                result['days'] = 1
            elif '后天' in query:
                result['days'] = 2
            else:
                result['days'] = QUERY_SETTINGS["default_forecast_days"]
        
        return result
    
    def format_current_weather(self, weather_data: Dict, city_name: str) -> str:
        """
        格式化当前天气信息
        
        Args:
            weather_data: 天气数据
            city_name: 城市名称
            
        Returns:
            格式化的天气信息字符串
        """
        try:
            temp = weather_data['main']['temp']
            feels_like = weather_data['main']['feels_like']
            humidity = weather_data['main']['humidity']
            description = weather_data['weather'][0]['description']
            wind_speed = weather_data['wind']['speed']
            
            result = f"📍 {city_name} 当前天气:\n"
            result += f"🌡️  温度: {temp:.1f}°C (体感 {feels_like:.1f}°C)\n"
            result += f"☁️  天气: {description}\n"
            result += f"💧  湿度: {humidity}%\n"
            result += f"💨  风速: {wind_speed} m/s\n"
            
            return result
            
        except KeyError as e:
            return f"天气数据格式错误: {e}"
    
    def format_forecast(self, forecast_data: Dict, city_name: str, days: int) -> str:
        """
        格式化天气预报信息
        
        Args:
            forecast_data: 预报数据
            city_name: 城市名称
            days: 预报天数
            
        Returns:
            格式化的预报信息字符串
        """
        try:
            result = f"📍 {city_name} 未来{days}天天气预报:\n\n"
            
            # 按日期分组
            daily_forecasts = {}
            for item in forecast_data['list'][:days * 8]:  # 每天8个时间点
                date = datetime.fromtimestamp(item['dt']).date()
                if date not in daily_forecasts:
                    daily_forecasts[date] = []
                daily_forecasts[date].append(item)
            
            for i, (date, forecasts) in enumerate(daily_forecasts.items()):
                if i >= days:
                    break
                    
                # 计算当天的平均温度和主要天气
                temps = [f['main']['temp'] for f in forecasts]
                descriptions = [f['weather'][0]['description'] for f in forecasts]
                
                min_temp = min(temps)
                max_temp = max(temps)
                main_desc = max(set(descriptions), key=descriptions.count)
                
                # 格式化日期
                if date == datetime.now().date():
                    date_str = "今天"
                elif date == datetime.now().date() + timedelta(days=1):
                    date_str = "明天"
                elif date == datetime.now().date() + timedelta(days=2):
                    date_str = "后天"
                else:
                    date_str = date.strftime("%m月%d日")
                
                result += f"📅 {date_str} ({date.strftime('%Y-%m-%d')}):\n"
                result += f"   🌡️  {min_temp:.1f}°C ~ {max_temp:.1f}°C\n"
                result += f"   ☁️  {main_desc}\n\n"
            
            return result
            
        except KeyError as e:
            return f"预报数据格式错误: {e}"
    
    def query_weather(self, natural_query: str) -> str:
        """
        处理自然语言天气查询
        
        Args:
            natural_query: 自然语言查询
            
        Returns:
            天气信息字符串
        """
        # 解析查询
        parsed = self.parse_natural_language(natural_query)
        
        if not parsed['city']:
            return "❌ 请指定要查询的城市名称，例如：'北京的天气如何？'"
        
        # 获取坐标
        coords = self.get_coordinates(parsed['city'])
        if not coords:
            return f"❌ 找不到城市 '{parsed['city']}'，请检查城市名称是否正确。"
        
        lat, lon = coords
        
        # 根据查询类型获取天气信息
        if parsed['query_type'] == 'current':
            weather_data = self.get_current_weather(lat, lon)
            if weather_data:
                return self.format_current_weather(weather_data, parsed['city'])
            else:
                return "❌ 获取天气信息失败，请稍后重试。"
        
        elif parsed['query_type'] == 'forecast':
            forecast_data = self.get_forecast(lat, lon, parsed['days'])
            if forecast_data:
                return self.format_forecast(forecast_data, parsed['city'], parsed['days'])
            else:
                return "❌ 获取天气预报失败，请稍后重试。"

def main():
    """
    主函数 - 交互式天气查询
    """
    print("🌤️  天气助手启动！")
    
    # 获取API密钥
    api_key = None
    
    if CONFIG_AVAILABLE and API_KEY:
        print("✅ 从配置文件读取API密钥")
        api_key = API_KEY
    else:
        print("请设置您的OpenWeatherMap API密钥")
        print("获取API密钥: https://openweathermap.org/api")
        print("提示: 您可以将API密钥保存在config.py文件中（参考config_example.py）")
        print()
        
        api_key = input("请输入您的API密钥: ").strip()
        if not api_key:
            print("❌ API密钥不能为空！")
            return
    
    # 创建天气助手实例
    assistant = WeatherAssistant(api_key)
    
    print("\n✅ 天气助手已就绪！")
    print("您可以这样询问:")
    print(f"  - {DEFAULT_CITY}的天气怎么样？")
    print("  - 上海明天的天气")
    print("  - 广州未来3天的天气预报")
    print("  - 输入 'quit' 或 'exit' 退出")
    print()
    
    while True:
        try:
            query = input("🤔 请问您想了解哪里的天气？ ").strip()
            
            if query.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            
            if not query:
                continue
            
            print("\n🔍 正在查询天气信息...")
            result = assistant.query_weather(query)
            print(f"\n{result}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()