from core.ir.prims.prim import Primitive, PrimitiveType
from basics.addr_blocks_in_mem import AddrBlocksInMem
from basics.data_block import DataBlock
from core.simulator.emulator.core import Core as EmuCore
from core.simulator.analyser.core import Core as AnalyCore
from core.simulator.analyser.event import *
from core.simulator.analyser.event import MsgEvents
from core.utils.tensor2intbv import tensor2intbv

from copy import deepcopy
from myhdl import bin, intbv
import warnings
from typing import Dict, List, Optional
import numpy as np

def to_signed_bits(val, width=16):
    if not -(1 << (width-1)) <= val < (1 << (width-1)):
        raise ValueError("超出范围: %d位有符号数" % width)
    return val & ((1 << width) - 1)

def to_unsigned_bits(val, width=16):
    if not 0 <= val < (1 << width):
        raise ValueError("超出范围: %d位无符号数" % width)
    return val & ((1 << width) - 1) 

def from_signed_bits(val, width=16):
    if val >= (1 << (width - 1)):
        return val - (1 << width)
    return val
        

class SendMsg():
    def __init__(self,
        S, 
        T,
        E,
        Q,
        LVDS: int = 0,
        Y: int = 0,
        X: int = 0,
        A0: int = 0,
        pack_per_rhead: int = 0,
        A_offset: int = 0,
        Const: int = 0,
        handshake: int = 0,
        tag_id: int = 0,
        en: int = 1,
        sparse: int = 0,
        ):
        self.S = int(S)
        self.T = int(T)
        self.E = int(E)
        self.Q = int(Q)
        self.LVDS = int(LVDS)
        self.Y = int(Y)
        self.X = int(X)
        self.A0 = int(A0)
        self.pack_per_rhead = int(pack_per_rhead)
        self.A_offset = int(A_offset)
        self.Const = int(Const)
        self.handshake = int(handshake)
        self.tag_id = int(tag_id)
        self.en = int(en)
        self.sparse = int(sparse)
        
        self.setPIC()

    def setPIC(self):
        self.PIC = intbv(0, min=0, max=(1<<128))
        self.PIC[0] = to_unsigned_bits(int(self.S), 1)
        self.PIC[1] = to_unsigned_bits(int(self.T), 1)
        self.PIC[2] = to_unsigned_bits(int(self.E), 1)
        self.PIC[3] = to_unsigned_bits(int(self.Q), 1)
        self.PIC[6:4] = to_unsigned_bits(int(self.LVDS), 2)
        self.PIC[12:6] = to_signed_bits(int(self.Y), 6)
        self.PIC[18:12] = to_signed_bits(int(self.X), 6)
        self.PIC[32:18] = to_signed_bits(int(self.A0), 14)
        self.PIC[44:32] = to_signed_bits(int(self.pack_per_rhead), 12)
        self.PIC[56:44] = to_signed_bits(int(self.A_offset), 12)
        self.PIC[63:56] = to_signed_bits(int(self.Const), 7)
        self.PIC[63] = to_unsigned_bits(int(self.handshake), 1)
        self.PIC[72:64] = to_signed_bits(int(self.tag_id), 8)
        self.PIC[72] = to_unsigned_bits(int(self.en), 1)
        self.PIC[73] = to_unsigned_bits(int(self.sparse), 1)

    def parsePIC(self, pic):
        """解析 PIC (128位 intbv或int) 并更新属性"""
        if isinstance(pic, int):
             self.PIC = intbv(pic, min=0, max=(1<<128))
        else:
             self.PIC = pic
        
        self.S = int(self.PIC[0])
        self.T = int(self.PIC[1])
        self.E = int(self.PIC[2])
        self.Q = int(self.PIC[3])
        self.LVDS = int(self.PIC[6:4])
        self.Y = from_signed_bits(int(self.PIC[12:6]), 6)
        self.X = from_signed_bits(int(self.PIC[18:12]), 6)
        self.A0 = from_signed_bits(int(self.PIC[32:18]), 14)
        self.pack_per_rhead = int(self.PIC[44:32])
        self.A_offset = from_signed_bits(int(self.PIC[56:44]), 12)
        self.Const = int(self.PIC[63:56])
        self.handshake = int(self.PIC[63])
        self.tag_id = int(self.PIC[72:64])
        self.en = int(self.PIC[72])
        self.sparse = int(self.PIC[73])

    def to_intbv(self):
        """返回配置好的 PIC (128位 intbv)"""
        return self.PIC


class PrimSendRecv(Primitive):
    def __init__(self,
        deps: int = 0b00000000,
        send: bool = False,
        send_addr: int = 0,
        cell_or_neuron: int = 0,
        pack_head_num: int = 0,
        router_table_addr: int = 0,
        recv: bool = False,
        recv_addr: int = 0,
        CXY: int = 0,
        mc_y: int = 0,
        mc_x: int = 0,
        tag_id: int = 0,
        ignore_end_count: int = 0,
        end_num: int = 0):
        
        super().__init__()
        self.name = "PrimSendRecv"
        self.deps = deps

        self.send_enable = bool(send)
        self.recv_enable = bool(recv)
        
        self.type = PrimitiveType.ROUTER

        if self.send_enable:
            self.send_addr = int(send_addr)
            self.cell_or_neuron = int(cell_or_neuron)  # 0: 8B cell, 1: 1B neuron
            self.pack_head_num = int(pack_head_num)    # message_num
            self.router_table_addr = int(router_table_addr)
            self.msg_list: List[SendMsg] = []
            # self.router_table = router_table
            # 路由表：每个32B里存放两个128b的message配置
            # if router_table is None:
            #     raise ValueError("router_table 不能为空 (DataBlock, addressing=\"32B\")")
            # if not isinstance(router_table, DataBlock):
            #     raise TypeError("router_table 必须是 DataBlock")
            # if router_table.addressing != "32B":
            #     raise ValueError("router_table 的addressing必须为 \"32B\"")
            # self.data_blocks["router_table"] = deepcopy(router_table)

        if self.recv_enable:
            self.recv_addr = int(recv_addr)
            self.CXY = int(CXY)  # 00/11: none, 01: multicast, 10: relay
            self.mc_y = int(mc_y)
            self.mc_x = int(mc_x)
            self.tag_id = int(tag_id)
            self.end_num = int(end_num)
            self.ignore_end_count = int(ignore_end_count)

        # 生成PIC
        self.setPIC()
        self.reset()
    
    def reset(self):
        if self.send_enable:
            self.message_count = 0
            self.cur_send_addr = self.send_addr if self.cell_or_neuron == 0 else self.send_addr << 5 # 32B to 1B

    def _gen_send_pic(self) -> intbv:
        pic = intbv(0, min=0, max=(1<<256))
        # mode/op
        pic[4:0] = 0x6
        pic[4] = 1
        pic[16:8] = to_unsigned_bits(int(self.deps), 8)

        # fields per prim_0x6_0x1_send.sv
        # send_addr @ [63:48]
        
        try:
            pic[64:48] = to_unsigned_bits(int(self.send_addr), 16)
        except Exception:
            pic[64:48] = 0  # 如果转换失败则设置为全0

        # cell_or_neuron @ [168]
        pic[168] = to_unsigned_bits(int(self.cell_or_neuron), 1)
        # send_rhead_num (message_num) @ [183:176]
        pic[184:176] = to_unsigned_bits(int(self.pack_head_num), 8)
        # router_table_addr @ [255:240]
        pic[256:240] = to_unsigned_bits(int(self.router_table_addr), 16)
        return pic

    def _gen_recv_pic(self) -> intbv:
        pic = intbv(0, min=0, max=(1<<256))
        # mode/op
        pic[4:0] = 0x6
        pic[5] = 1
        # deps
        pic[16:8] = to_unsigned_bits(int(self.deps), 8)

        # recv_addr @ [47:32]
        pic[48:32] = to_unsigned_bits(int(self.recv_addr), 16)
        # relay/CXY @ [173:172], relay_x @ [197:192], relay_y @ [189:184]
        pic[169] = to_unsigned_bits(int(self.ignore_end_count), 1)
        pic[174:172] = to_unsigned_bits(int(self.CXY), 2)
        pic[198:192] = to_signed_bits(int(self.mc_x), 6)
        pic[190:184] = to_signed_bits(int(self.mc_y), 6)
        # tag_id @ [207:200], end_num @ [215:208]
        pic[208:200] = to_unsigned_bits(int(self.tag_id), 8)
        pic[216:208] = to_unsigned_bits(int(self.end_num), 8)
        return pic

    def setPIC(self):
        # 同时支持Send与Recv：当两者都有效时，组合单一统一PIC（置位[24]和[25]）
        pic = intbv(0, min=0, max=(1<<256))
        self.PIC_send = intbv(0, min=0, max=(1<<256))
        self.PIC_recv = intbv(0, min=0, max=(1<<256))
        if self.send_enable:
            self.PIC_send = self._gen_send_pic()
            pic |= self.PIC_send
        if self.recv_enable:
            self.PIC_recv = self._gen_recv_pic()
            pic |= self.PIC_recv
        self.PIC = pic

    def setRouterAddr(self,router_table_addr):
        self.router_table_addr = router_table_addr
        self.PIC[256:240] = to_unsigned_bits(int(router_table_addr), 16)
      
    def setdeps(self, deps):
        self.deps = deps
        self.PIC[16:8] = to_unsigned_bits(int(deps), 8)
    
    def execute(self, core: EmuCore):
        self.y, self.x = core.pos_y, core.pos_x
        
        if self.recv_enable:
            core.receiving = True
            core.tag = self.tag_id
            core.recv_addr = self.recv_addr
            core.recv_num = self.end_num + 1
            core.CXY = self.CXY
            core.mc_y = self.mc_y
            core.mc_x = self.mc_x
        
        if self.send_enable:
            core.sending = True
            # 计算消息数
            message_len = (self.pack_head_num + 1 + 1) // 2
            input_message = core.memory[self.router_table_addr : self.router_table_addr + message_len]
            
            # 读取输入message
            cur_message_cell = input_message[self.message_count // 2]
            cur_message = SendMsg(0,0,0,0)
            if self.message_count % 2 == 0:
                cur_message.parsePIC(tensor2intbv(cur_message_cell[0:16]))
            else:
                cur_message.parsePIC(tensor2intbv(cur_message_cell[16:32]))
            
            dest_y, dest_x = self.y + cur_message.Y, self.x + cur_message.X
            
            # message无效
            if cur_message.en == 0:
                self.message_count += 1
                
                if self.message_count > self.pack_head_num:
                    core.sending = False
                    self.reset()
                    
                return
            
            # 不握手情况
            if not cur_message.handshake:
                if not core.noc[(dest_y, dest_x)].receiving:
                    warnings.warn("Core (%d, %d) to Core (%d, %d) 无握手，发送数据可能丢失" % (self.y, self.x, dest_y, dest_x))
                    return

            if cur_message.Q and core.noc[(dest_y, dest_x)].CXY == 1:  # 多播
                # multicast
                mc_y = core.noc[(dest_y, dest_x)].mc_y + dest_y
                mc_x = core.noc[(dest_y, dest_x)].mc_x + dest_x
                if not core.noc[(mc_y, mc_x)].receiving:
                    warnings.warn("多播核 Core (%d, %d) ，多播无握手，发送数据可能丢失" % (mc_y, mc_x))
                    return
            
            # 握手情况
            if not core.noc[(dest_y, dest_x)].receiving:
                return
            if core.noc[(dest_y, dest_x)].tag != cur_message.tag_id:
                return
            
            # send data
            def send_cell():
                cell_num = cur_message.pack_per_rhead + 1
                recv_addr_8B = (core.noc[(dest_y, dest_x)].recv_addr << 2) + cur_message.A0
                interval = 0
                offset = cur_message.A_offset
                for i in range(cell_num):
                    input_cell = core.memory[self.cur_send_addr]
                    
                    for j in range(4):
                        input_tensor = input_cell[j*8:(j+1)*8]
                        core.noc[(dest_y, dest_x)].memory.write8B(recv_addr_8B, input_tensor)
                        
                        if cur_message.Q and core.noc[(dest_y, dest_x)].CXY == 1:  # 多播
                            mc_y = core.noc[(dest_y, dest_x)].mc_y + dest_y
                            mc_x = core.noc[(dest_y, dest_x)].mc_x + dest_x
                            mc_recv_addr_8B = recv_addr_8B - (core.noc[(dest_y, dest_x)].recv_addr << 2) + (core.noc[(mc_y, mc_x)].recv_addr << 2)
                            core.noc[(mc_y, mc_x)].memory.write8B(mc_recv_addr_8B, input_tensor)
                        
                        recv_addr_8B += 1
                    
                    recv_addr_8B += (offset - 1) if interval == (cur_message.Const) else 0
                    interval = (interval + 1) % (cur_message.Const + 1)
                    
                    self.cur_send_addr += 1
                
                core.noc[(dest_y, dest_x)].recv_num -= 1
                if core.noc[(dest_y, dest_x)].recv_num == 0:
                    core.noc[(dest_y, dest_x)].receiving = False
                    core.noc[(dest_y, dest_x)].tag = None
                    core.noc[(dest_y, dest_x)].recv_addr = None
                
                if cur_message.Q and core.noc[(dest_y, dest_x)].CXY == 1:  # 多播
                    mc_y = core.noc[(dest_y, dest_x)].mc_y + dest_y
                    mc_x = core.noc[(dest_y, dest_x)].mc_x + dest_x
                    core.noc[(mc_y, mc_x)].recv_num -= 1
                    if core.noc[(mc_y, mc_x)].recv_num == 0:
                        core.noc[(mc_y, mc_x)].receiving = False
                        core.noc[(mc_y, mc_x)].tag = None
                        core.noc[(mc_y, mc_x)].recv_addr = None
                        
            def send_neuron():
                neuron_num = cur_message.pack_per_rhead + 1
                recv_addr_1B = core.noc[(dest_y, dest_x)].recv_addr << 5 + cur_message.A0
                interval = 0
                offset = cur_message.A_offset
                for i in range(neuron_num):
                    input_neuron = core.memory.read1B(self.cur_send_addr)
                    core.noc[(dest_y, dest_x)].memory.write1B(recv_addr_1B, input_neuron)
                    
                    if cur_message.Q and core.noc[(dest_y, dest_x)].CXY == 1:  # 多播
                        mc_y = core.noc[(dest_y, dest_x)].mc_y + dest_y
                        mc_x = core.noc[(dest_y, dest_x)].mc_x + dest_x
                        mc_recv_addr_1B = recv_addr_1B - (core.noc[(dest_y, dest_x)].recv_addr << 5) + (core.noc[(mc_y, mc_x)].recv_addr << 5)
                        core.noc[(mc_y, mc_x)].memory.write1B(mc_recv_addr_1B, input_neuron)
                    
                    recv_addr_1B += (offset - 1) if interval == (cur_message.Const) else 0
                    interval = (interval + 1) % (cur_message.Const + 1)
                    
                    self.cur_send_addr += 1
                
                core.noc[(dest_y, dest_x)].recv_num -= 1
                if core.noc[(dest_y, dest_x)].recv_num == 0:
                    core.noc[(dest_y, dest_x)].receiving = False
                    core.noc[(dest_y, dest_x)].tag = None
                    core.noc[(dest_y, dest_x)].recv_addr = None
                
                if cur_message.Q and core.noc[(dest_y, dest_x)].CXY == 1:  # 多播
                    mc_y = core.noc[(dest_y, dest_x)].mc_y + dest_y
                    mc_x = core.noc[(dest_y, dest_x)].mc_x + dest_x
                    core.noc[(mc_y, mc_x)].recv_num -= 1
                    if core.noc[(mc_y, mc_x)].recv_num == 0:
                        core.noc[(mc_y, mc_x)].receiving = False
                        core.noc[(mc_y, mc_x)].tag = None
                        core.noc[(mc_y, mc_x)].recv_addr = None
                
            
            if self.cell_or_neuron == 0:
                send_cell()
            else:
                send_neuron()
            
            # 更新发送消息计数
            self.message_count += 1
            if self.message_count > self.pack_head_num:
                core.sending = False
                self.reset()
                
    
    def generate_events(self, core: AnalyCore):
        events: List[MsgEvents] = []
        
        self.y, self.x = core.core_pos
        
        if self.recv_enable:
            # 只有一个CommEvent，和一个WriteMemoryEvent
            
            recv_event = CommEvent(
                name=f"Recv_{self.tag_id}",
                parent=core.router.full_name,
                volume=0,  # 体积未知，后续根据实际接收数据量更新
                max_bandwidth=1,  # 带宽未知，后续根据实际情况更新
                energy=0,  # 能耗未知，后续根据实际情况更新
                sending=False,
                
                CXY=self.CXY,
                mc_y=self.mc_y,
                mc_x=self.mc_x,
                recv_num=self.end_num + 1,
                tag=self.tag_id
            )
            recv_event.remaining_time = float('inf')  # 接收事件持续到所有数据接收完成，时间未知，先设置为无穷大
            
            msg_events = MsgEvents(
                sync_event=None,
                router_event=recv_event,
                mem_events=[None]
            )
            
            events.append(msg_events)
        
        if self.send_enable:
                
            for cur_message in self.msg_list:
            
                dest_y, dest_x = self.y + cur_message.Y, self.x + cur_message.X
                
                # message无效
                if cur_message.en == 0:
                    continue
                
                sync_event = SyncEvent(
                    name=f"Sync_{cur_message.tag_id}",
                    parent=core.router.full_name,
                    position=(self.y, self.x),
                    sync_targets=[(dest_y, dest_x)],
                    sending=True,
                    energy=core.config['noc']['energy_per_hop'] * 2 * (48/96.0) * (abs(cur_message.Y) + abs(cur_message.X)),  # 能耗未知，后续根据实际情况更新
                    tag=cur_message.tag_id,
                    handshake=bool(cur_message.handshake),
                    Q=bool(cur_message.Q)
                )
                
                send_energy = 0
                if self.cell_or_neuron == 0:
                    send_energy = core.config['noc']['energy_per_hop'] * (cur_message.pack_per_rhead + 1) * 4 * (abs(cur_message.Y) + abs(cur_message.X))  # 简单估算能耗
                else:
                    send_energy = core.config['noc']['energy_per_hop'] * (cur_message.pack_per_rhead + 1) * (40/96.0) * (abs(cur_message.Y) + abs(cur_message.X))  # 简单估算能耗
                
                send_event = CommEvent(
                    name=f"Send_{cur_message.tag_id}",
                    parent=core.router.full_name,
                    volume=(cur_message.pack_per_rhead + 1) * (32 if self.cell_or_neuron == 0 else 1),
                    max_bandwidth=core.config['noc']['normal_router_width'] if self.cell_or_neuron == 0 else core.config['noc']['neuron_router_width'],  
                    energy=send_energy,
                    sending=True,
                    target=(dest_y, dest_x),
                    Q=bool(cur_message.Q),
                    cell_or_neuron=self.cell_or_neuron
                )
                
                total_cycle = (cur_message.pack_per_rhead + 1) * (4 if self.cell_or_neuron == 0 else 1)
                total_volumn = (cur_message.pack_per_rhead + 1) * (32 if self.cell_or_neuron == 0 else 1)  # 包括message
                
                read_event = MemoryEvent(
                    name=f"Send_read",
                    parent=core.memory[0].full_name,
                    memory_type=EventType.READ,
                    volume=(cur_message.pack_per_rhead + 1) * (32 if self.cell_or_neuron == 0 else 1),
                    bounded_events=[send_event.full_name],
                    energy=np.ceil(total_volumn / core.config["core"]["memory_width"]) * core.config["core"]["L0_memory_read_energy"],  # 能耗未知，后续根据实际情况更新
                    max_bandwidth=  total_volumn / total_cycle  # 带宽未知，后续根据实际情况更新
                )
                
                msg_events = MsgEvents(
                    sync_event=sync_event,
                    router_event=send_event,
                    mem_events=[read_event]
                )
                events.append(msg_events)
        
        return events