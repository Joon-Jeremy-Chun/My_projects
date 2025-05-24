# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:49:02 2025

@author: joonc
"""

from djitellopy import Tello

# 드론 객체 생성
tello = Tello()

# 드론 연결 시도
tello.connect()

# 연결 상태 및 배터리 상태 출력
print(f"연결 성공! 배터리 상태: {tello.get_battery()}%")
