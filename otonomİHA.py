import sys
import asyncio
import time
import os
import json

from mavsdk import System, telemetry, offboard, gimbal, gimbal_pb2, action
from mavsdk.gimbal import GimbalMode, ControlMode, GimbalError
from mavsdk.tune import (SongElement, TuneDescription, TuneError)

"""
export PX4_HOME_LAT=39.9530363 
export PX4_HOME_LON=32.5824773
export PX4_HOME_ALT=0
cd Firameware
make px4_sitl_default jmavsim
"""

#!/usr/bin/env python3

async def run():
    print("Run started")
    drone = System()
    #await drone.connect(system_address="serial://COM6:115200")
    #await drone.connect(system_address="serial://COM10:57600")
    await drone.connect(system_address="serial:///dev/ttyACM0:115200")
    #await drone.connect(system_address="udp://:14540")
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone discovered!")
            break
    await drone.action.set_takeoff_altitude(2)
    await drone.action.set_return_to_launch_altitude(5)
    fileName="temp"
    print(fileName)
    otonom = await drone.mission_raw.import_qgroundcontrol_mission(fileName)
    await drone.mission_raw.upload_mission(otonom.mission_items)
    await drone.mission.set_return_to_launch_after_mission(True)
    print("Waiting for drone to have a global position estimate...")
    """async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            print("Global position estimate ok")
            break
    print("Fetching amsl altitude at home location....")
    async for terrain_info in drone.telemetry.home():
        absolute_altitude = terrain_info.absolute_altitude_m
        break"""
    async for is_armed in drone.telemetry.armed():
        time.sleep(0.2)
        print("Waiting for ARM")
        if (is_armed):
            print("ARMED")
            break
    print("-- Starting mission")
    await drone.mission.start_mission()
    await asyncio.sleep(300)
    await drone.action.return_to_launch()
    print("RTL Enabled")
if __name__ == "__main__":
    items = []
    def start():
        start_message = {
            "autoContinue": True,
            "command": 530,
            "doJumpId": 0,
            "frame": 2,
            "params": [
                0,
                2,
                None,
                None,
                None,
                None,
                None,
            ],
            "type": "SimpleItem"
        }
        items.append(start_message)
    def exit(jump):
        exit_messsage = {
            "autoContinue": True,
            "command": 177,
            "doJumpId": jump,
            "frame": 2,
            "params": [
                jump,
                20,
                0,
                0,
                0,
                0,
                0
            ],
            "type": "SimpleItem"
        }
        items.append(exit_messsage)
    def go_to(lat, lon, alt, speed, jump):
        # commend 178
        speed_message = {
            "autoContinue": True,
            "command": 178,
            "doJumpId": jump,
            "frame": 2,
            "params": [
                1,
                speed,
                -1,
                0,
                0,
                0,
                0
            ],
            "type": "SimpleItem"
        }
        # commend 16
        go_message = {
            "AMSLAltAboveTerrain": 5,
            "Altitude": alt,
            "AltitudeMode": 1,
            "autoContinue": True,
            "command": 16,
            "doJumpId": jump + 1,
            "frame": 3,
            "params": [
                0,
                0,
                0,
                None,
                lat,
                lon,
                alt
            ],
            "type": "SimpleItem"
        }
        items.append(speed_message)
        items.append(go_message)
    def set_roi(lat, lon, alt, jump):
        # commend 195
        roi_message = {
            "AMSLAltAboveTerrain": 5,
            "Altitude": 5,
            "AltitudeMode": 1,
            "autoContinue": True,
            "command": 195,
            "doJumpId": jump,
            "frame": 3,
            "params": [
                0,
                0,
                0,
                0,
                lat,
                lon,
                alt
            ],
            "type": "SimpleItem"
        }
        items.append(roi_message)

    def missionRTL(doJumpIdInput):
        rtlMessage = {
            "autoContinue": True,
            "command": 20,
            "doJumpId": doJumpIdInput,
            "frame": 2,
            "params": [
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "type": "SimpleItem"
        }
        items.append(rtlMessage)

    def missionDelay(delayTime, doJumpIdInput):
        missionDelayMessage = {
            "autoContinue": True,
            "command": 93,
            "doJumpId": doJumpIdInput,
            "frame": 2,
            "params": [
                delayTime,
                0,
                0,
                0,
                0,
                0,
                0
            ],
            "type": "SimpleItem"
        }
        items.append(missionDelayMessage)
    json_message = {"fileType": "Plan", "geoFence": {"circles": [
    ],
        "polygons": [
        ],
        "version": 2
    },
                    "groundStation": "QGroundControl",
                    "mission": {
                        "cruiseSpeed": 15,
                        "firmwareType": 12,
                        "globalPlanAltitudeMode": 1,
                        "hoverSpeed": 2,
                        "items": items
                        ,
                        "plannedHomePosition": [
                            39.9530592,
                            32.5824274,
                            809.9737600000287
                        ],
                        "vehicleType": 2,
                        "version": 2
                    },
                    "rallyPoints": {
                        "points": [
                        ],
                        "version": 2
                    },
                    "version": 1
                    }
    start()
    #---İlk Alt Dört Köşe---
    go_to(39.8946186, 32.7871321, 2, 2, 1)    #1
    go_to(39.8946541, 32.7871287, 2, 2, 3)     #2
    go_to(39.8946515, 32.7870824, 2, 2, 5)    #3
    go_to(39.8946161, 32.7870843, 2, 2, 7)    #4
    go_to(39.8946186, 32.7871321, 2, 2, 9)    #1

    #---İlk Yükseliş
    go_to(39.8946186, 32.7871321, 6, 2, 11)    #1
    go_to(39.8946541, 32.7871287, 6, 2, 13)    #2
    go_to(39.8946541, 32.7871287, 2, 2, 15)    #2 alçalış
    go_to(39.8946541, 32.7871287, 6, 2, 17)    #2 yükseliş
    go_to(39.8946515, 32.7870824, 6, 2, 19)    #3
    go_to(39.8946515, 32.7870824, 2, 2, 21)    #3 alçalış
    go_to(39.8946515, 32.7870824, 6, 2, 23)    #3 yükseliş
    go_to(39.8946161, 32.7870843, 6, 2, 25)    #4
    go_to(39.8946161, 32.7870843, 2, 2, 27)    #4 alçalış
    go_to(39.8946161, 32.7870843, 6, 2, 29)    #4 Yükseliş
    go_to(39.8946186, 32.7871321, 6, 2, 31)    # 1
    missionDelay(3, 33)

    #---ucgen---
    go_to(39.8946186, 32.7871321, 5, 2, 34)   # 1 5metre
    set_roi(39.8946270, 32.7871071, 5, 36)
    go_to(39.8946475, 32.7871044, 5, 2, 37)
    go_to(39.8946161, 32.7870843, 5, 2, 39)
    go_to(39.8946186, 32.7871321, 5, 2, 41)
    exit(36)

    json_dict = json.dumps(json_message)
    print(json_dict)
    f = open("temp", "w")
    f.write(json_dict)
    f.close()
    time.sleep(5)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    os.remove("temp")