# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:58:43 2025

@author: joonc
"""

from djitellopy import Tello
import time

# 드론 객체 생성 및 연결
tello = Tello()
tello.connect()

# 현재 배터리 상태 확인 (중요!)
battery = tello.get_battery()
print(f"현재 배터리 상태: {battery}%")

# 안전 비행을 위한 배터리 체크 (20% 미만이면 중단)
if battery < 20:
    print("배터리가 부족합니다! 비행을 취소합니다.")
else:
    # 이륙
    tello.takeoff()
    print("🚀 드론이 이륙했습니다!")

    # 공중에서 5초간 대기
    time.sleep(5)

    # 착륙
    tello.land()
    print("🛬 드론이 착륙했습니다!")
