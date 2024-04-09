// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__PACKET_HPP_
#define RM_SERIAL_DRIVER__PACKET_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rm_serial_driver
{
struct ReceivePacket
{
  uint8_t header = 0x5A;
  uint8_t detect_color;  // 0-red 1-blue
  uint8_t mode;  // 
  //bool reset_tracker;
  uint8_t reserved;
  //float roll;
  //float pitch;
  //float yaw;
  //float aim_x;
  //float aim_y;
  //float aim_z;
  int16_t keepr;
  uint16_t checksum = 0;
};// __attribute__((packed));

struct SendPacket
{
  uint8_t header = 0xA5;
  //bool tracking;
  uint8_t id;          // 0-outpost 6-guard 7-base
  uint8_t armors_num;  // 2-balance 3-outpost 4-normal
  uint8_t reserved;
  float x;
  float y;
  float z;
  float yaw;
  float pitch;
  float roll;
  //float vz;
  //float v_yaw;
  //float r1;
  //float r2;
  //float dz;
  uint16_t keeps;  // complement
  uint16_t checksum = 0;
};// __attribute__((packed));

inline ReceivePacket fromVector(const std::vector<uint8_t> & data)
{
  ReceivePacket packet;
  std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&packet));
  return packet;
}

inline std::vector<uint8_t> toVector(const SendPacket & data)
{
  std::vector<uint8_t> packet(sizeof(SendPacket));
  std::copy(
    reinterpret_cast<const uint8_t *>(&data),
    reinterpret_cast<const uint8_t *>(&data) + sizeof(SendPacket), packet.begin());
  return packet;
}

}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__PACKET_HPP_
