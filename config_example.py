#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件示例
将此文件复制为 config.py 并填入您的API密钥
"""

# OpenWeatherMap API密钥
# 获取地址: https://openweathermap.org/api
API_KEY = "your_api_key_here"

# 默认城市（可选）
DEFAULT_CITY = "北京"

# API设置
API_SETTINGS = {
    "base_url": "http://api.openweathermap.org/data/2.5",
    "geocoding_url": "http://api.openweathermap.org/geo/1.0",
    "units": "metric",  # 温度单位：metric(摄氏度), imperial(华氏度), kelvin(开尔文)
    "lang": "zh_cn"     # 语言：zh_cn(中文), en(英文)
}

# 查询设置
QUERY_SETTINGS = {
    "max_forecast_days": 5,  # 最大预报天数
    "default_forecast_days": 3  # 默认预报天数
}