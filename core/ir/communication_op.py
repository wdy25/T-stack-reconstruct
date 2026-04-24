from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from myhdl import bin, intbv
from core.ir.data import MemBlock, Data, ViewData


from core.ir.prims.send_recv_prim import PrimSendRecv, SendMsg

class CommOp(ABC):

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.attrs = attrs or {}

    
class SendOp(CommOp):
    '''
    Send operation for communication.
    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary.
            Expected keys in attrs:
                - 'dest' (str): Destination address or identifier.
                - 'tag' (int): Message tag for identifying the message type.
    '''

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['source', 'dest', 'tag']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Send operation.")
        self.primitive = True  # Marking as a non-primitive operation
        # 手动配置的标志和存储
        self.manual_config = False
        self.manual_send_prim_config = None
        self.manual_msg_configs = None
    
    def configure_send_manual(self, send_prim_config: Dict[str, Any], msg_configs: List[Dict[str, Any]]) -> None:
        self.manual_config = True
        self.manual_send_prim_config = send_prim_config
        self.manual_msg_configs = msg_configs
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        max_8B_size = 4096
        max_1B_size = 4096 / 32 
        #Self msg的列表 2^(14 -2) = 4096个
        
        if(len(inputs) >1):
            raise ValueError(f"Only 1 input is allowed now.")
        elif(inputs[0].memref.length > max_8B_size):
            raise ValueError(f"Input data size is too large.")
        # elif(inputs[0].memref % 8 != 0):
        #     raise ValueError(f"Input data size is not divisible by 8.")

        # 检查是否使用手动配置
        if self.manual_config:
            # 使用手动配置生成 Send 原语
            send_prim_cfg = self.manual_send_prim_config.copy()
            if 'deps' not in send_prim_cfg:
                send_prim_cfg['deps'] = 0
            if 'send' not in send_prim_cfg:
                send_prim_cfg['send'] = True
            if 'send_addr' not in send_prim_cfg:
                send_prim_cfg['send_addr'] = inputs[0].memref.addr if inputs[0].memref.addr is not None else -1
            if 'router_table_addr' not in send_prim_cfg:
                send_prim_cfg['router_table_addr'] = 0
            
            self.send_prim = PrimSendRecv(**send_prim_cfg)
            
            # 使用手动配置生成 Msg 列表
            self.msg_list = []
            for msg_cfg in self.manual_msg_configs:
                msg = SendMsg(**msg_cfg)
                self.msg_list.append(msg)
        else:
            # 使用默认配置生成 Send 原语
            self.send_prim = PrimSendRecv(
                deps=0,
                send=True,
                send_addr=inputs[0].memref.addr if inputs[0].memref.addr is not None else 0,
                cell_or_neuron=0,
                pack_head_num=0,
                router_table_addr=0,
                # router_table=None
            )
            
            # 使用默认配置生成 Msg 列表
            self.msg_list = []
            msg = SendMsg(
                S=1,
                T=0,
                E=0,
                Q=0,
                LVDS=0,
                Y=self.attrs["dest"][0] - self.attrs["source"][0],
                X=self.attrs["dest"][1] - self.attrs["source"][1],
                A0=0,
                pack_per_rhead=inputs[0].memref.length-1,
                A_offset=1,
                Const=0,
                handshake=1,
                tag_id=self.attrs['tag'],
                en=1,
                sparse=0
            )
            self.msg_list.append(msg)
    
        # Combine messages into memory blocks (128-bit msg -> 256-bit memblock)
        # Each memblock can hold 2 messages
        memblocks = []
        for i in range(0, len(self.msg_list), 2):
            combined_payload = intbv(0, min=0, max=(1<<256))
            if i + 1 < len(self.msg_list):
                # Two messages available, combine them
                # msg1 = self.msg_list[i]
                # msg2 = self.msg_list[i + 1]
                # Pack two 128-bit messages into one 256-bit memblock
                # combined_payload = (msg2.to_intbv() << 128) | msg1.to_intbv()
                combined_payload[128:0] = self.msg_list[i].to_intbv()
                combined_payload[255:128] = self.msg_list[i + 1].to_intbv()
            else:
                # Only one message available, pad with zeros
                # msg1 = self.msg_list[i]
                combined_payload[128:0] = self.msg_list[i].to_intbv()
                combined_payload[255:128] = intbv(0, min=0, max=(1<<128))
                # msg2 = intbv(0, min=0, max=(1<<128))
                # msg2 = 0  # 128 bits of zeros
                # Pad the high 128 bits with zeros
                # combined_payload = (msg2 << 128) | msg1.to_intbv()
            
            memblocks.append(combined_payload)
        
        # Create parameter memory block
        para_code = MemBlock(length=len(memblocks), payload=memblocks)
        para_data = Data(name=f"{self.name}.params", memref=para_code)
        return para_data

    def build_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        if self.manual_config:
            send_prim_cfg = self.manual_send_prim_config.copy()
            send_prim_cfg['deps'] = 0
            send_prim_cfg['send'] = True
            send_prim_cfg['send_addr'] = inputs[0].memref.addr
            send_prim_cfg['router_table_addr'] = 0
            self.send_prim = PrimSendRecv(**send_prim_cfg)
            
            para_data = inputs[1]
            self.send_prim.setRouterAddr(para_data.memref.addr)
            self.send_prim.setdeps(deps)
        else:  
            self.send_prim = PrimSendRecv(
                deps=0,
                send=True,
                send_addr=inputs[0].memref.addr,
                cell_or_neuron=0,
                pack_head_num=0,
                router_table_addr=0
            )

            para_data = inputs[1]
            self.send_prim.setRouterAddr(para_data.memref.addr)
            self.send_prim.setdeps(deps)
        self.send_prim.msg_list = self.msg_list
        return self.send_prim
        
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        max_8B_size = 4096
        max_1B_size = 4096 / 32 


        if self.manual_config:
            send_prim_cfg = self.manual_send_prim_config.copy()
            send_prim_cfg['deps'] = 0
            send_prim_cfg['send'] = True
            send_prim_cfg['send_addr'] = inputs[0].memref.addr
            send_prim_cfg['router_table_addr'] = 0
            self.send_prim = PrimSendRecv(**send_prim_cfg)
            
            para_data = inputs[1]
            self.send_prim.setRouterAddr(para_data.memref.addr)
            self.send_prim.setdeps(deps)
            return self.send_prim.PIC
        else:  
            self.send_prim = PrimSendRecv(
                deps=0,
                send=True,
                send_addr=inputs[0].memref.addr,
                cell_or_neuron=0,
                pack_head_num=0,
                router_table_addr=0
            )

            para_data = inputs[1]
            self.send_prim.setRouterAddr(para_data.memref.addr)
            self.send_prim.setdeps(deps)
            return self.send_prim.PIC

    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
        
class RecvOp(CommOp):
    '''
    Receive operation for communication.
    Attributes:
        name (str): Human-readable identifier for the operation.
        attrs (Dict[str, Any]): Operation parameters dictionary.
            Expected keys in attrs:
                - 'source' (str): Source address or identifier.
                - 'tag' (int): Message tag for identifying the message type.
    '''

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['source', 'tag']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Recv operation.")
        self.primitive = True  # Marking as a non-primitive operation
        
        # 手动配置的标志和存储
        self.manual_config = False
        self.manual_recv_prim_config = None

    def configure_recv_manual(self, recv_prim_config: Dict[str, Any]) -> None:

        self.manual_config = True
        self.manual_recv_prim_config = recv_prim_config

    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        return None
    
    def build_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        # 检查是否使用手动配置
        if self.manual_config:
            # 使用手动配置生成 Recv 原语
            recv_prim_cfg = self.manual_recv_prim_config.copy()
            recv_prim_cfg['deps'] = deps
            recv_prim_cfg['recv'] = True
            recv_prim_cfg['recv_addr'] = outputs[0].memref.addr
            
            self.recv_prim = PrimSendRecv(**recv_prim_cfg)
        else:
            # 使用默认配置生成 Recv 原语
            self.recv_prim = PrimSendRecv(
                deps=deps,
                recv=True,
                recv_addr=outputs[0].memref.addr,
                CXY=0,
                mc_y=0,
                mc_x=0,
                tag_id=self.attrs['tag'],
                end_num=0
            )
        return self.recv_prim

    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        
        # 检查是否使用手动配置
        if self.manual_config:
            # 使用手动配置生成 Recv 原语
            recv_prim_cfg = self.manual_recv_prim_config.copy()
            recv_prim_cfg['deps'] = deps
            recv_prim_cfg['recv'] = True
            recv_prim_cfg['recv_addr'] = outputs[0].memref.addr
            
            self.recv_prim = PrimSendRecv(**recv_prim_cfg)
        else:
            # 使用默认配置生成 Recv 原语
            self.recv_prim = PrimSendRecv(
                deps=deps,
                recv=True,
                recv_addr=outputs[0].memref.addr,
                CXY=0,
                mc_y=0,
                mc_x=0,
                tag_id=self.attrs['tag'],
                end_num=0
            )
        return self.recv_prim.PIC

    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return False
    