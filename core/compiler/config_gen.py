from typing import Dict, List, Tuple, Any, Optional, Set
from myhdl import bin, intbv
from pathlib import Path

from core.ir.graph import Graph, NodeId
from core.ir.hardware_graph import HardwareGraph
from core.ir.data import Data, ViewData, MemBlock
from core.ir.operation import Operation
from core.ir.communication_op import CommOp
from core.ir.control_op import ControlOp
from core.ir.prims.send_recv_prim import PrimSendRecv
from core.ir.prims.stop import PrimStop


class ConfigGenerator:
    def __init__(self, hg: HardwareGraph, codes: Dict[Tuple[int,int], List[intbv]], output_dir: Path = Path("./temp"), settings: Dict[str, Any] = {}, core_shift: Tuple[int, int] = (0, 0), loop_connection:bool = False) -> None:
        self.hg = hg
        self.codes = codes
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if "simple" not in settings:
            settings["simple"] = False
        if "continuous" not in settings:
            settings["continuous"] = True
        if "prim_base_addr" not in settings:
            settings["prim_base_addr"] = 0x0
        if "xy_order" not in settings:
            settings["xy_order"] = False
        if "ls_control_permission" not in settings:
            settings["ls_control_permission"] = True
        if "dendrite_clk_gating" not in settings:
            settings["dendrite_clk_gating"] = True
        if "soma_clk_gating" not in settings:
            settings["soma_clk_gating"] = True
        if "move_clk_gating" not in settings:
            settings["move_clk_gating"] = True
        if "lvds_parallel" not in settings:
            settings["lvds_parallel"] = True
        if "lvds_calib" not in settings:
            settings["lvds_calib"] = 0
        if "send_mode" not in settings:
            settings["send_mode"] = 1
        if "polling_times" not in settings:
            settings["polling_times"] = 1000
        if "performance_measurement" not in settings:
            settings["performance_measurement"] = True
        if "start_all" not in settings:
            settings["start_all"] = False
        
        self.settings = settings
        
        self.core_mapping: Dict[Tuple[int, int], Tuple[int, int]] = {}  # from original core_id to shifted core_id in the die
        self.core_group: Dict[Tuple[int, int], Tuple[int, int]] = {}  # from original core_id to die id
        self.core_shift = core_shift
        self.loop_connection = loop_connection
        
        core_id_set = set(self.codes.keys())
        if loop_connection:  # x和y的最大间距都必须小于4
            max_y = max([core_id[0] for core_id in core_id_set])
            min_y = min([core_id[0] for core_id in core_id_set])
            max_x = max([core_id[1] for core_id in core_id_set])
            min_x = min([core_id[1] for core_id in core_id_set])
            assert max_y - min_y < 4 and max_x - min_x < 4, "For loop connection, the max distance of x and y coordinates of cores must be less than 4"
        
        for core_id in core_id_set:            
            y_shifted = core_id[0] + core_shift[0]
            x_shifted = core_id[1] + core_shift[1] 
            
            die_id = (y_shifted // 4, x_shifted // 4) if not loop_connection else (0, 0)
            core_id_in_die = (y_shifted % 4, x_shifted % 4)
            self.core_mapping[core_id] = core_id_in_die
            self.core_group[core_id] = die_id
            
        self.one_die = len(set(self.core_group.values())) == 1
        if not self.one_die:  
            # 创建文件夹
            for die_id in set(self.core_group.values()):
                die_output_dir = self.output_dir.joinpath(f"die_{die_id[0]}_{die_id[1]}")
                die_output_dir.mkdir(parents=True, exist_ok=True)
        

    def generate_ctrl_config(self, dump: bool = True, filename: str = None) -> Dict[Tuple[int,int], str]:
        configs: Dict[Tuple[int,int], str] = {}
        for core_id in self.codes:
            config_str = ""
            
            reg_code = intbv(0)[31:]
            reg_code[0] = 0
            reg_code[1] = 1 if self.settings["simple"] else 0
            reg_code[2] = 1 if self.settings["continuous"] else 0
            reg_code[17:3] = self.settings["prim_base_addr"]
            reg_code[18] = 1 if self.settings["xy_order"] else 0
            config_str += f"@80008 {reg_code}\n"
            
            reg_code = intbv(0)[31:]
            reg_code[0] = 0
            reg_code[1] = 1 if self.settings["ls_control_permission"] else 0
            config_str += f"@80010 {reg_code}\n"
            
            reg_code = intbv(0)[31:]
            reg_code[0] = 1 if self.settings["dendrite_clk_gating"] else 0
            reg_code[1] = 1 if self.settings["soma_clk_gating"] else 0
            reg_code[2] = 1 if self.settings["move_clk_gating"] else 0
            config_str += f"@8001c {reg_code}\n"
            
            if dump is True:
                if self.one_die:
                    with open(self.output_dir.joinpath(f"{self.core_mapping[core_id][0]}_{self.core_mapping[core_id][1]}_ctrl_{filename if filename is not None else 'config'}.txt"), "w") as f:
                        f.write(config_str)
                else:
                    die_id = self.core_group[core_id]
                    die_output_dir = self.output_dir.joinpath(f"die_{die_id[0]}_{die_id[1]}")
                    with open(die_output_dir.joinpath(f"{self.core_mapping[core_id][0]}_{self.core_mapping[core_id][1]}_ctrl_{filename if filename is not None else 'config'}.txt"), "w") as f:
                        f.write(config_str)
            
        return configs
    
    def generate_mem_config(self, prim_base_addr: Optional[int] = None, dump: bool = True, filename: str = None) -> Dict[Tuple[int,int], str]:
        if prim_base_addr is None:
            prim_base_addr = self.settings.get("prim_base_addr", 0x0)
        configs: Dict[Tuple[int,int], str] = {}
        for core_id in self.codes:
            config_str = ""
            # for prim codes
            for i, code in enumerate(self.codes[core_id]):
                config_str += f"@{(prim_base_addr + i):04X} {code}\n"

            code_length = len(self.codes[core_id])
            
            # for data blocks
            core_nodes = set(self.hg.get_nodes_by_core(core_id))
            for i, nid in enumerate(core_nodes):
                if self.hg.kind_of(nid) == "data":
                    data_node = self.hg.node(nid)
                    assert isinstance(data_node, Data)
                    assert data_node.memref is not None, f"{nid}: Data node must have a memory reference"
                    mem_block: MemBlock = data_node.memref
                    assert mem_block.addr is not None and mem_block.addr >= 0, f"{nid}: MemBlock must have a valid address"
                    if mem_block.payload is None or len(mem_block.payload) == 0:
                        continue
                    assert mem_block.addr + len(mem_block.payload) <= prim_base_addr or prim_base_addr + code_length <= mem_block.addr, f"{nid}: Memory blocks must not overlap with prim codes"
                    for j, data in enumerate(mem_block.payload):
                        config_str += f"@{(mem_block.addr + j):04X} {data}\n"
            configs[core_id] = config_str
            if dump is True:
                if self.one_die:
                    with open(self.output_dir.joinpath(f"{self.core_mapping[core_id][0]}_{self.core_mapping[core_id][1]}_mem_{filename if filename is not None else 'config'}.txt"), "w") as f:
                        f.write(config_str)
                else:
                    die_id = self.core_group[core_id]
                    die_output_dir = self.output_dir.joinpath(f"die_{die_id[0]}_{die_id[1]}")
                    with open(die_output_dir.joinpath(f"{self.core_mapping[core_id][0]}_{self.core_mapping[core_id][1]}_mem_{filename if filename is not None else 'config'}.txt"), "w") as f:
                        f.write(config_str)
        return configs
    
    def generate_core_list(self, dump: bool = True) -> str:
        if self.one_die:
            core_list_str = ""
            for core_id in self.codes:
                core_list_str += f"{self.core_mapping[core_id][0]} {self.core_mapping[core_id][1]}\n"
            if dump is True:
                with open(self.output_dir.joinpath(f"core_list.txt"), "w") as f:
                    f.write(core_list_str)
            return core_list_str
        else:
            # write die list
            die_list_str = ""
            for die_id in set(self.core_group.values()):
                die_list_str += f"{die_id[0]} {die_id[1]}\n"
            if dump is True:
                with open(self.output_dir.joinpath(f"die_list.txt"), "w") as f:
                    f.write(die_list_str)
                    
            # write core list
            core_list_str: Dict[Tuple[int, int], str] = {}
            for core_id in self.codes:
                die_id = self.core_group[core_id]
                if die_id not in core_list_str:
                    core_list_str[die_id] = ""
                core_list_str[die_id] += f"{self.core_mapping[core_id][0]} {self.core_mapping[core_id][1]}\n"
            if dump is True:
                for die_id in core_list_str:
                    die_output_dir = self.output_dir.joinpath(f"die_{die_id[0]}_{die_id[1]}")
                    with open(die_output_dir.joinpath(f"core_list.txt"), "w") as f:
                        f.write(core_list_str[die_id])
                        
    def generate_spi_config(self, dump: bool = True) -> Dict[int, str]:
        if self.one_die:
            pass
        else:
            # write die list
            die_list_str = ""
            for die_id in set(self.core_group.values()):
                die_list_str += f"{die_id[0]} {die_id[1]}\n"
            if dump is True:
                with open(self.output_dir.joinpath(f"die_list.txt"), "w") as f:
                    f.write(die_list_str)
            
            for die_id in set(self.core_group.values()):
                die_y, die_x = die_id
                die_spi_file = self.output_dir.joinpath(f"die_{die_y}_{die_x}").joinpath(f"_spi_config.txt")
                spi_config_str = ""
                for core_id in self.codes:
                    if self.core_group[core_id] != die_id:
                        continue
                    core_y, core_x = self.core_mapping[core_id]
                    
    def generate_lvds_config(self, dump: bool = True, base_core_id: Tuple[int, int] = (0,0), use_end_num: bool = True):
        prim_base_addr = self.settings.get("prim_base_addr", 0x0)
        
        core_id_set = set(self.codes.keys())
        lvds_config: Dict[Tuple[int, int], Any] = {}
        for core_id in core_id_set:                        
            mem_cfg_0: Dict[int, intbv] = {}
            mem_cfg_1: Dict[int, intbv] = {}
            mem_cfg_2: Dict[int, intbv] = {}
            mem_cfg_3: Dict[int, intbv] = {}
            
            code_length = len(self.codes[core_id])
            
            # for data blocks
            core_nodes = set(self.hg.get_nodes_by_core(core_id))
            for i, nid in enumerate(core_nodes):
                if self.hg.kind_of(nid) == "data":
                    data_node = self.hg.node(nid)
                    assert isinstance(data_node, Data)
                    assert data_node.memref is not None, f"{nid}: Data node must have a memory reference"
                    mem_block: MemBlock = data_node.memref
                    assert mem_block.addr is not None and mem_block.addr >= 0, f"{nid}: MemBlock must have a valid address"
                    if mem_block.payload is None or len(mem_block.payload) == 0:
                        continue
                    assert mem_block.addr + len(mem_block.payload) <= prim_base_addr or prim_base_addr + code_length <= mem_block.addr, f"{nid}: Memory blocks must not overlap with prim codes"
                    for j, data in enumerate(mem_block.payload):
                        addr = mem_block.addr + j
                        if addr < 0x1000: # 0-128KB
                            mem_cfg_0[addr] = data
                        elif addr < 0x2000: # 128KB-256KB
                            mem_cfg_1[addr] = data
                        elif addr < 0x3000: # 256KB-384KB
                            mem_cfg_2[addr] = data
                        else: # 384KB-512KB
                            assert addr < 0x4000, f"Address {addr} out of range for LVDS config"
                            mem_cfg_3[addr] = data

            # 计算真实运行的原语起始地址
            prim_shift = int(len(mem_cfg_0) > 0) + int(len(mem_cfg_1) > 0) + int(len(mem_cfg_2) > 0) + int(len(mem_cfg_3) > 0) + 1
            assert prim_shift < 0x1000
            
            # 生成接收原语
            recv_prims = []
            if len(mem_cfg_0) > 0:
                pp = PrimSendRecv(
                    deps=0b00000000,
                    send=False,
                    recv=True,
                    recv_addr=prim_base_addr + prim_shift,
                    ignore_end_count=(not use_end_num),
                    end_num=0
                )
                recv_prims.append(pp.PIC)
            if len(mem_cfg_1) > 0:
                pp = PrimSendRecv(
                    deps=0b00000001,
                    send=False,
                    recv=True,
                    recv_addr=0x1000,
                    ignore_end_count=(not use_end_num),
                    end_num=0
                )
                recv_prims.append(pp.PIC)
            if len(mem_cfg_2) > 0:
                pp = PrimSendRecv(
                    deps=0b00000001,
                    send=False,
                    recv=True,
                    recv_addr=0x2000,
                    ignore_end_count=(not use_end_num),
                    end_num=0
                )
                recv_prims.append(pp.PIC)
            if len(mem_cfg_3) > 0:
                pp = PrimSendRecv(
                    deps=0b00000001,
                    send=False,
                    recv=True,
                    recv_addr=0x3000,
                    ignore_end_count=(not use_end_num),
                    end_num=0
                )
                recv_prims.append(pp.PIC)
            st = PrimStop(
                deps=0b00000001,
                jump_addr=0,
                jump=0,
                relative=0
            )
            recv_prims.append(st.PIC)
            
            # 生成接收原语配置文件
            config_str = ""
            # for prim codes
            for i, code in enumerate(recv_prims):
                config_str += f"@{(prim_base_addr + i):04X} {code}\n"
            
            filename = None
            if dump is True:
                if self.one_die:
                    with open(self.output_dir.joinpath(f"{self.core_mapping[core_id][0]}_{self.core_mapping[core_id][1]}_mem_{filename if filename is not None else 'config'}.txt"), "w") as f:
                        f.write(config_str)
                else:
                    die_id = self.core_group[core_id]
                    die_output_dir = self.output_dir.joinpath(f"die_{die_id[0]}_{die_id[1]}")
                    with open(die_output_dir.joinpath(f"{self.core_mapping[core_id][0]}_{self.core_mapping[core_id][1]}_mem_{filename if filename is not None else 'config'}.txt"), "w") as f:
                        f.write(config_str)
            
            # 写入主程序原语
            for i, code in enumerate(self.codes[core_id]):
                assert prim_base_addr + prim_shift + i not in mem_cfg_0, f"Address conflict at {prim_base_addr + prim_shift + i:04X}"
                assert prim_base_addr + prim_shift + i < 0x1000
                mem_cfg_0[prim_base_addr + prim_shift + i] = code
            
            lvds_config[core_id] = {}
            if len(mem_cfg_0) > 0:
                lvds_config[core_id][prim_base_addr + prim_shift] = mem_cfg_0
            if len(mem_cfg_1) > 0:
                lvds_config[core_id][0x1000] = mem_cfg_1
            if len(mem_cfg_2) > 0:
                lvds_config[core_id][0x2000] = mem_cfg_2
            if len(mem_cfg_3) > 0:
                lvds_config[core_id][0x3000] = mem_cfg_3
            
        # 将每个核的原语和数据转换为lvds配置文件
        # 根据核坐标从远到近排序
        sorted_core_ids = sorted(lvds_config.keys(), key=lambda cid: (abs((cid[0] + self.core_shift[0]) - (base_core_id[0])), abs((cid[1] + self.core_shift[1]) - (base_core_id[1]))), reverse=True)
        
        def mem2lvds(addr: int, data: intbv, delta_y: int, delta_x: int) -> str:
            assert 0 <= addr < 0x1000, f"Address {addr} out of range for LVDS config"
            lvds_str = ""
            lvds_str += f"00,{delta_y},{delta_x},0,0,{addr<<2},1,{data[64:0]}\n"
            lvds_str += f"00,{delta_y},{delta_x},0,0,{addr<<2|0x1},1,{data[128:64]}\n"
            lvds_str += f"00,{delta_y},{delta_x},0,0,{addr<<2|0x2},1,{data[192:128]}\n"
            lvds_str += f"00,{delta_y},{delta_x},0,0,{addr<<2|0x3},1,{data[256:192]}\n"
            return lvds_str     
        
        def lvds_end(delta_y: int, delta_x: int, end_num: int = 0) -> str:
            # assert 0 <= addr < 0x1000, f"Address {addr} out of range for LVDS config"
            lvds_str = ""
            lvds_str += f"00,{delta_y},{delta_x},1,{end_num},0,1,0\n"
            return lvds_str         
        
        config_str = ""
        for core_id in sorted_core_ids:
            y_shifted = core_id[0] + self.core_shift[0]
            x_shifted = core_id[1] + self.core_shift[1] 
            
            delta_y = y_shifted - (base_core_id[0])
            delta_x = x_shifted - (base_core_id[1])
            
            mem_cfgs = lvds_config[core_id]
            for base_addr, mem_cfg in mem_cfgs.items():
                packet_num = len(mem_cfg) * 4
                for addr in sorted(mem_cfg.keys()):
                    config_str += mem2lvds(addr - base_addr, mem_cfg[addr], delta_y, delta_x)
                config_str += lvds_end(delta_y, delta_x, packet_num)
        
        if dump is True:
            with open(self.output_dir.joinpath(f"lvds_config.txt"), "w") as f:
                f.write(config_str)            
  
    def generate_all_configs(self) -> None:
        self.generate_ctrl_config()
        self.generate_mem_config()
        self.generate_core_list()
        
        if self.one_die:
            generate_c_codes(
                settings=self.settings,
                core_list=[self.core_mapping[core_id] for core_id in self.codes],
                path=self.output_dir,
                filename="test.c"
            )
        else:
            for die_id in set(self.core_group.values()):
                die_output_dir = self.output_dir.joinpath(f"die_{die_id[0]}_{die_id[1]}")
                core_list_in_die = [self.core_mapping[core_id] for core_id in self.codes if self.core_group[core_id] == die_id]
                generate_c_codes(
                    settings=self.settings,
                    core_list=core_list_in_die,
                    path=die_output_dir,
                    filename="test.c"
                )
    
    def generate_lvds_c_configs(self, base_core_id: Tuple[int, int] = (0,0)) -> None:
        self.generate_lvds_config(base_core_id=base_core_id)
        self.generate_core_list()
        
        if self.one_die:
            generate_complete_c_codes(
                settings=self.settings,
                core_list=[self.core_mapping[core_id] for core_id in self.codes],
                path=self.output_dir,
                filename="test.c"
            )
        else:
            for die_id in set(self.core_group.values()):
                die_output_dir = self.output_dir.joinpath(f"die_{die_id[0]}_{die_id[1]}")
                core_list_in_die = [self.core_mapping[core_id] for core_id in self.codes if self.core_group[core_id] == die_id]
                generate_complete_c_codes(
                    settings=self.settings,
                    core_list=core_list_in_die,
                    path=die_output_dir,
                    filename="test.c"
                )

def generate_complete_c_codes(settings: Dict[str, Any], core_list: List[Tuple[int, int]], path: Path, filename: str) -> str:

    # Defaults
    if "simple" not in settings:
        settings["simple"] = False
    if "continuous" not in settings:
        settings["continuous"] = True
    if "prim_base_addr" not in settings:
        settings["prim_base_addr"] = 0x0
    if "xy_order" not in settings:
        settings["xy_order"] = False
    if "ls_control_permission" not in settings:
        settings["ls_control_permission"] = True
    if "dendrite_clk_gating" not in settings:
        settings["dendrite_clk_gating"] = True
    if "soma_clk_gating" not in settings:
        settings["soma_clk_gating"] = True
    if "move_clk_gating" not in settings:
        settings["move_clk_gating"] = True
    if "lvds_parallel" not in settings:
        settings["lvds_parallel"] = True
    if "lvds_calib" not in settings:
        settings["lvds_calib"] = 0
    if "send_mode" not in settings:
        settings["send_mode"] = 1
    if "polling_times" not in settings:
        settings["polling_times"] = 1000
    if "performance_measurement" not in settings:
        settings["performance_measurement"] = True
    if "start_all" not in settings:
        settings["start_all"] = False

    # Ensure Path object and output dir exists
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Helpers to convert settings to bit literals used in C expressions
    b = lambda v: 1 if bool(v) else 0
    prim_base_addr_val = int(settings.get("prim_base_addr", 0))

    # Build C array for used cores
    pairs = ", ".join(f"{{{y}, {x}}}" for (y, x) in core_list)
    core_list_array = f"int core_list[][2] = {{{pairs}}};\n    int core_count = sizeof(core_list)/sizeof(core_list[0]);\n"

    # Optional LVDS calibration block
    lvds_calib_block = ""
    if int(settings.get("lvds_calib", 0)) != 0:
        lvds_calib_block = (
            "    // if LVDS calib needed\n"
            "    write_reg32(0x01160100, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n"
            "    write_reg32(0x01160110, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n"
            "    write_reg32(0x01160120, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n"
            "    write_reg32(0x01160130, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n\n"
        )
    
    # Optional performance measurement code
    perf_measure_block0 = ""
    perf_measure_block1 = ""
    perf_measure_block2 = ""
    perf_measure_block00 = ""
    if settings.get("performance_measurement"):
        perf_measure_block0 = (
            "    uint32_t clk_cnt = 0;\n"
        )
        perf_measure_block00 = (
            "    clk_cnt = 0;\n"
        )
        
        perf_measure_block1 = (
            "            else {\n"
            "                uint32_t rdata = read_core_reg(0x02080020);\n"
            "                if (rdata > clk_cnt) clk_cnt = rdata;\n"
            "            }\n"
        )
        perf_measure_block2 = (
            f"    if (loop_count <= {settings.get('polling_times', 10)}) {{\n"
            "        printf(\"All cores idle detected in %u loops. Max clk count: %u\\n\", loop_count, clk_cnt);\n"
            "    }\n"
        )
    
    start_all_block = ""
    start_one_block = ""
    if settings.get("start_all", False):
        start_all_block = (
            "    // 启动计算（广播启动脉冲）\n"
            "    write_reg32(0x02fff000,(0x1<<8|0x1));\n"
            "    write_reg32(0x02fff00c,(0x0<<2|0x0));\n"
            "    write_reg32(0x02080004, (0x1)); // start\n"
            "    write_reg32(0x02080004, (0x0)); // start\n\n"
        )
        start_one_block = ""
    else:
        start_all_block = ""
        start_one_block = (
            "        write_reg32(0x02080004, (0x1)); // start\n"
            "        write_reg32(0x02080004, (0x0)); // start"
        )

    # # Escape braces in the core_list array for .format
    # core_list_array_escaped = core_list_array.replace('{', '{{').replace('}', '}}')

    # Build the C source using a format template; double braces are literal C braces
    c_template = """
#include "datatype.h"
#include "common.h"

uint32_t read_core_reg(uint32_t addr) {{
    uint32_t rdata;
    rdata = read_reg32(addr);
    while(1) {{
        rdata= read_reg32(0x02fff008);
        if(rdata &(0x1<<1)) break;
    }}
    write_reg32(0x02fff008,(0x1<<1|0x0));
    rdata = read_reg32(0x02fff010);
    return rdata;
}}

int main (void) {{
    printf("\\n---\\n");

    // LVDS 广播写入使能
    write_reg32(0x01160400, (0x1<<8|0x1<<0));

    // LVDS parallel disable off on 4 lanes
    write_reg32(0x01160000, (0x{send_mode:x}<<2|0x{lvds_direct:x}<<1|0x{lvds_direct:x}<<0));
    write_reg32(0x01160004, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));
    write_reg32(0x01160024, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));
    write_reg32(0x01160044, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));
    write_reg32(0x01160064, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));

{lvds_calib_block}    // 关闭 LVDS 写入使能
    write_reg32(0x01160400, (0x0<<8|0x0<<0));

    // 配置各个核共用的参数（广播）
    write_reg32(0x02fff000,(0x1<<8|0x1));  // [8]广播写使能, [0]读写使能
    write_reg32(0x02fff00c,(0x0<<2|0x0));  // 广播写入所有核

    write_reg32(0x0208000c, 0x0); // cfg_clk off
    write_reg32(0x02080010, (0x{ls_perm:x}<<1|0x0)); // [1]ls_control_permission, [0]ls
    write_reg32(0x0208001c,(0x{move:x}<<2|0x{soma:x}<<1|0x{dend:x})); // move/soma/dendrite clg enable
    write_reg32(0x02080000, 0x1);         // 释放所有rst

    // 只对使用到的核进行配置
    {core_list_array}
    for(int i = 0; i < core_count; ++i) {{
        int y = core_list[i][0];
        int x = core_list[i][1];
        write_reg32(0x02fff000,(0x0<<8|0x1)); // 关闭广播, 开启读写
        write_reg32(0x02fff00c,(y<<2|x));     // 选择单核 (y, x)
        write_reg32(0x02080008, (0x{xy:x}<<18|0x{base:x}<<3|0x{cont:x}<<2|0x{simple:x}<<1|0x1)); // config_en=1
        write_reg32(0x02080008, (0x{xy:x}<<18|0x{base:x}<<3|0x{cont:x}<<2|0x{simple:x}<<1|0x0)); // config_en=0
{start_one_block}
    }}

{start_all_block}

    // 轮询使用到的核，等待全部 idle
    uint32_t all_cores_idle=0;
    uint32_t loop_count=0;
{perf_measure_block0}
    while(!all_cores_idle){{
        all_cores_idle=1;
        for(int i = 0; i < core_count; ++i) {{
            int y = core_list[i][0];
            int x = core_list[i][1];
            write_reg32(0x02fff000,(0x0<<8|0x1));
            write_reg32(0x02fff00c,(y<<2|x));
            uint32_t rdata= read_core_reg(0x02080018);
            if(rdata==0) {{
                all_cores_idle=0;
                break;
            }}
{perf_measure_block1}
        }}
        loop_count++;
        if(loop_count > {polling_times}) {{
            printf("CASE FAILED: Recving Polling timeout!\\n");
            break;
        }}
    }}
{perf_measure_block2}

{start_all_block}

    for(int i = 0; i < core_count; ++i) {{
        int y = core_list[i][0];
        int x = core_list[i][1];
        write_reg32(0x02fff000,(0x0<<8|0x1)); // 关闭广播, 开启读写
        write_reg32(0x02fff00c,(y<<2|x));     // 选择单核 (y, x)
{start_one_block}
    }}

    all_cores_idle=0;
    loop_count=0;
{perf_measure_block00}
    // 轮询使用到的核，等待全部 idle
    while(!all_cores_idle){{
        all_cores_idle=1;
        for(int i = 0; i < core_count; ++i) {{
            int y = core_list[i][0];
            int x = core_list[i][1];
            write_reg32(0x02fff000,(0x0<<8|0x1));
            write_reg32(0x02fff00c,(y<<2|x));
            uint32_t rdata= read_core_reg(0x02080018);
            if(rdata==0) {{
                all_cores_idle=0;
                break;
            }}
{perf_measure_block1}
        }}
        loop_count++;
        if(loop_count > {polling_times}) {{
            printf("CASE FAILED: Polling timeout!\\n");
            break;
        }}
    }}
{perf_measure_block2}

    return 0;
}}
""".strip()

    c_source = c_template.format(
        lvds_direct=0 if settings.get('lvds_parallel') else 1,
        send_mode=b(settings.get('send_mode')),
        lvds_calib_block=lvds_calib_block,
        ls_perm=b(settings.get('ls_control_permission')),
        move=b(settings.get('move_clk_gating')),
        soma=b(settings.get('soma_clk_gating')),
        dend=b(settings.get('dendrite_clk_gating')),
        xy=b(settings.get('xy_order')),
        base=prim_base_addr_val,
        cont=b(settings.get('continuous')),
        simple=b(settings.get('simple')),
        core_list_array=core_list_array,
        polling_times=settings.get('polling_times', 10),
        perf_measure_block0=perf_measure_block0,
        perf_measure_block1=perf_measure_block1,
        perf_measure_block2=perf_measure_block2,
        perf_measure_block00=perf_measure_block00,
        start_one_block=start_one_block,
        start_all_block=start_all_block
    )

    # Write to file
    out_file = path.joinpath(filename)
    with open(out_file, "w") as f:
        f.write(c_source)

    return c_source
        

def generate_c_codes(settings: Dict[str, Any], core_list: List[Tuple[int, int]], path: Path, filename: str) -> str:
    """Generate a C source file similar to temp/demo.c.

    The generated code:
    - Initializes LVDS and optional calibration.
    - Broadcasts common registers.
    - Iterates only over `core_list` to configure and release rst per used core.
    - Starts computation via broadcast start pulse.
    - Polls only used cores for idle via `read_core_reg`.

    Args:
        settings: Control knobs (simple, continuous, prim_base_addr, xy_order,
                  ls_control_permission, dendrite_clk_gating, soma_clk_gating,
                  move_clk_gating, lvds_calib).
        core_list: List of (y, x) cores to configure and poll.
        path: Output directory to write the C file into.

    Returns:
        The generated C source as a string.
    """
    # Defaults
    if "simple" not in settings:
        settings["simple"] = False
    if "continuous" not in settings:
        settings["continuous"] = True
    if "prim_base_addr" not in settings:
        settings["prim_base_addr"] = 0x0
    if "xy_order" not in settings:
        settings["xy_order"] = False
    if "ls_control_permission" not in settings:
        settings["ls_control_permission"] = True
    if "dendrite_clk_gating" not in settings:
        settings["dendrite_clk_gating"] = True
    if "soma_clk_gating" not in settings:
        settings["soma_clk_gating"] = True
    if "move_clk_gating" not in settings:
        settings["move_clk_gating"] = True
    if "lvds_parallel" not in settings:
        settings["lvds_parallel"] = True
    if "lvds_calib" not in settings:
        settings["lvds_calib"] = 0
    if "send_mode" not in settings:
        settings["send_mode"] = 1
    if "polling_times" not in settings:
        settings["polling_times"] = 10
    if "performance_measurement" not in settings:
        settings["performance_measurement"] = True
    if "start_all" not in settings:
        settings["start_all"] = False

    # Ensure Path object and output dir exists
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Helpers to convert settings to bit literals used in C expressions
    b = lambda v: 1 if bool(v) else 0
    prim_base_addr_val = int(settings.get("prim_base_addr", 0))

    # Build C array for used cores
    pairs = ", ".join(f"{{{y}, {x}}}" for (y, x) in core_list)
    core_list_array = f"int core_list[][2] = {{{pairs}}};\n    int core_count = sizeof(core_list)/sizeof(core_list[0]);\n"

    # Optional LVDS calibration block
    lvds_calib_block = ""
    if int(settings.get("lvds_calib", 0)) != 0:
        lvds_calib_block = (
            "    // if LVDS calib needed\n"
            "    write_reg32(0x01160100, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n"
            "    write_reg32(0x01160110, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n"
            "    write_reg32(0x01160120, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n"
            "    write_reg32(0x01160130, (0x100<<16|0x16<<8|0x0<<4|0x1<<2|0x0<<1|0x0<<0)); // LVDS calib\n\n"
        )
    
    # Optional performance measurement code
    perf_measure_block0 = ""
    perf_measure_block1 = ""
    perf_measure_block2 = ""
    if settings.get("performance_measurement"):
        perf_measure_block0 = (
            "    uint32_t clk_cnt = 0;\n"
        )
        
        perf_measure_block1 = (
            "            else {\n"
            "                uint32_t rdata = read_core_reg(0x02080020);\n"
            "                if (rdata > clk_cnt) clk_cnt = rdata;\n"
            "            }\n"
        )
        perf_measure_block2 = (
            f"    if (loop_count <= {settings.get('polling_times', 10)}) {{\n"
            "        printf(\"All cores idle detected in %u loops. Max clk count: %u\\n\", loop_count, clk_cnt);\n"
            "    }\n"
        )
    
    start_all_block = ""
    start_one_block = ""
    if settings.get("start_all", False):
        start_all_block = (
            "    // 启动计算（广播启动脉冲）\n"
            "    write_reg32(0x02fff000,(0x1<<8|0x1));\n"
            "    write_reg32(0x02fff00c,(0x0<<2|0x0));\n"
            "    write_reg32(0x02080004, (0x1)); // start\n"
            "    write_reg32(0x02080004, (0x0)); // start\n\n"
        )
        start_one_block = ""
    else:
        start_all_block = ""
        start_one_block = (
            "        write_reg32(0x02080004, (0x1)); // start\n"
            "        write_reg32(0x02080004, (0x0)); // start"
        )

    # # Escape braces in the core_list array for .format
    # core_list_array_escaped = core_list_array.replace('{', '{{').replace('}', '}}')

    # Build the C source using a format template; double braces are literal C braces
    c_template = """
#include "datatype.h"
#include "common.h"

uint32_t read_core_reg(uint32_t addr) {{
    uint32_t rdata;
    rdata = read_reg32(addr);
    while(1) {{
        rdata= read_reg32(0x02fff008);
        if(rdata &(0x1<<1)) break;
    }}
    write_reg32(0x02fff008,(0x1<<1|0x0));
    rdata = read_reg32(0x02fff010);
    return rdata;
}}

int main (void) {{
    printf("\\n---\\n");

    // LVDS 广播写入使能
    write_reg32(0x01160400, (0x1<<8|0x1<<0));

    // LVDS parallel disable off on 4 lanes
    write_reg32(0x01160000, (0x{send_mode:x}<<2|0x{lvds_direct:x}<<1|0x{lvds_direct:x}<<0));
    write_reg32(0x01160004, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));
    write_reg32(0x01160024, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));
    write_reg32(0x01160044, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));
    write_reg32(0x01160064, (0x0<<15|0x8<<10|0x8<<5|0x1<<4|0x1<<3|0x1<<2|0x1<<1|0x1));

{lvds_calib_block}    // 关闭 LVDS 写入使能
    write_reg32(0x01160400, (0x0<<8|0x0<<0));

    // 配置各个核共用的参数（广播）
    write_reg32(0x02fff000,(0x1<<8|0x1));  // [8]广播写使能, [0]读写使能
    write_reg32(0x02fff00c,(0x0<<2|0x0));  // 广播写入所有核

    write_reg32(0x0208000c, 0x0); // cfg_clk off
    write_reg32(0x02080010, (0x{ls_perm:x}<<1|0x0)); // [1]ls_control_permission, [0]ls
    write_reg32(0x0208001c,(0x{move:x}<<2|0x{soma:x}<<1|0x{dend:x})); // move/soma/dendrite clg enable
    write_reg32(0x02080000, 0x1);         // 释放所有rst

    // 只对使用到的核进行配置
    {core_list_array}
    for(int i = 0; i < core_count; ++i) {{
        int y = core_list[i][0];
        int x = core_list[i][1];
        write_reg32(0x02fff000,(0x0<<8|0x1)); // 关闭广播, 开启读写
        write_reg32(0x02fff00c,(y<<2|x));     // 选择单核 (y, x)
        write_reg32(0x02080008, (0x{xy:x}<<18|0x{base:x}<<3|0x{cont:x}<<2|0x{simple:x}<<1|0x1)); // config_en=1
        write_reg32(0x02080008, (0x{xy:x}<<18|0x{base:x}<<3|0x{cont:x}<<2|0x{simple:x}<<1|0x0)); // config_en=0
{start_one_block}
    }}

{start_all_block}

    // 轮询使用到的核，等待全部 idle
    uint32_t all_cores_idle=0;
    uint32_t loop_count=0;
{perf_measure_block0}
    while(!all_cores_idle){{
        all_cores_idle=1;
        for(int i = 0; i < core_count; ++i) {{
            int y = core_list[i][0];
            int x = core_list[i][1];
            write_reg32(0x02fff000,(0x0<<8|0x1));
            write_reg32(0x02fff00c,(y<<2|x));
            uint32_t rdata= read_core_reg(0x02080018);
            if(rdata==0) {{
                all_cores_idle=0;
                break;
            }}
{perf_measure_block1}
        }}
        loop_count++;
        if(loop_count > {polling_times}) {{
            printf("CASE FAILED: Polling timeout!\\n");
            break;
        }}
    }}
{perf_measure_block2}
    return 0;
}}
""".strip()

    c_source = c_template.format(
        lvds_direct=0 if settings.get('lvds_parallel') else 1,
        send_mode=b(settings.get('send_mode')),
        lvds_calib_block=lvds_calib_block,
        ls_perm=b(settings.get('ls_control_permission')),
        move=b(settings.get('move_clk_gating')),
        soma=b(settings.get('soma_clk_gating')),
        dend=b(settings.get('dendrite_clk_gating')),
        xy=b(settings.get('xy_order')),
        base=prim_base_addr_val,
        cont=b(settings.get('continuous')),
        simple=b(settings.get('simple')),
        core_list_array=core_list_array,
        polling_times=settings.get('polling_times', 10),
        perf_measure_block0=perf_measure_block0,
        perf_measure_block1=perf_measure_block1,
        perf_measure_block2=perf_measure_block2,
        start_one_block=start_one_block,
        start_all_block=start_all_block
    )

    # Write to file
    out_file = path.joinpath(filename)
    with open(out_file, "w") as f:
        f.write(c_source)

    return c_source
        

def convert_mem_format(mem_data_file_in: str, mem_data_file_out: str, base_addr: int = 0x0, fill_gaps: bool = False) -> str:
    """Convert memory data file from 32B-addressed, 32B-per-line hex to 1B-addressed, 16B-per-line bytes.

    Input format example (32B addressing, one 32-byte line, hex packed, MSB..LSB):
        @0010 3ea9bf803f97bea0...000000002d2d2d0a

    Output format (1B addressing, 16 bytes per line, bytes low->high):
        @00000000 0A 2D 2D 2D 00 00 00 00 ... (16 bytes)
        @00000010 ... next 16 bytes ...

    Notes:
    - The input line's address is in units of 32 bytes ("32B寻址").
    - The output line's address is in units of 1 byte ("1B寻址"), and each
      output line contains 16 bytes. For an input address A (in 32B units),
      the corresponding byte address range is
      base_addr + (A - min_A) * 32 .. base_addr + (A - min_A) * 32 + 31.
    - Byte order in input is MSB..LSB; output requires low->high address order,
      so we reverse by byte (2 hex chars) before splitting into 16B lines.

    Args:
        mem_data_file_in: Path to the 32B-addressed input file.
        mem_data_file_out: Path to write the 1B-addressed output file.
        base_addr: Base byte address to add to the normalized output addresses.

    Returns:
        The converted content as a single string.
    """
    lines_out: list[str] = []

    def parse_line(line: str) -> tuple[int, str] | None:
        s = line.strip()
        if not s or not s.startswith('@'):
            return None
        # split at first whitespace after address
        try:
            after_at = s[1:]
            # find first space or tab
            sep_idx = None
            for i, ch in enumerate(after_at):
                if ch.isspace():
                    sep_idx = i
                    break
            if sep_idx is None:
                return None
            addr_hex = after_at[:sep_idx]
            data_hex = after_at[sep_idx:].strip().replace(" ", "").replace("\t", "")
            if not addr_hex or not data_hex:
                return None
            addr = int(addr_hex, 16)
            # normalize hex string characters
            data_hex = data_hex.lower()
            # ensure even number of hex digits
            if len(data_hex) % 2 != 0:
                raise ValueError(f"Data hex length not even at @{addr_hex}")
            return addr, data_hex
        except Exception:
            return None

    min_addr = None
    max_addr = None
    mem_data = {}

    with open(mem_data_file_in, 'r') as fin:
        for raw in fin:
            parsed = parse_line(raw)
            if parsed is None:
                continue
            addr32, data_hex = parsed
            mem_data[addr32] = data_hex
            if min_addr is None or addr32 < min_addr:
                min_addr = addr32
            if max_addr is None or addr32 > max_addr:
                max_addr = addr32
    
    if mem_data:
        if fill_gaps:
            addrs_to_process = range(min_addr, max_addr + 1)
        else:
            addrs_to_process = mem_data.keys()

        for addr32 in addrs_to_process:
            data_hex = mem_data.get(addr32, "00" * 32)

            # Split into bytes (2 hex chars each) and reverse to low->high address
            bytes_be = [data_hex[i:i+2] for i in range(0, len(data_hex), 2)]
            # If length is not 32 bytes, still handle generically, chunk in 16B groups
            bytes_le = list(reversed(bytes_be))

            # Generate 16B lines with 1B addressing (addr_out in bytes)
            for k in range(0, len(bytes_le), 16):
                chunk = bytes_le[k:k+16]
                if not chunk:
                    continue
                # compute byte-addressed starting address of this 16B chunk
                addr_byte = (addr32 - min_addr) * 32 + k + base_addr
                addr_str = f"@{addr_byte:08X}"
                byte_str = " ".join(b.upper() for b in chunk)
                lines_out.append(f"{addr_str} {byte_str}")
                # addr16 += 1

    out_str = "\n".join(lines_out) + ("\n" if lines_out else "")
    with open(mem_data_file_out, 'w') as fout:
        fout.write(out_str)

    return out_str


def combine_mem_configs(mem_data_files_in: List[str], mem_data_file_out: str) -> str:
    """Concatenate multiple mem config files in order without blank lines between.

    Behavior:
    - Reads each input file in the order provided.
    - Skips empty/whitespace-only lines.
    - Appends lines to the output sequentially with single newlines between lines.
    - Does NOT insert extra blank lines between files.
    - Writes the combined content to `mem_data_file_out` and returns it.

    Args:
        mem_data_files_in: List of input mem file paths in desired order.
        mem_data_file_out: Output file path for the combined mem content.

    Returns:
        The combined content as a single string.
    """
    lines_out: list[str] = []

    for in_path in mem_data_files_in:
        with open(in_path, 'r') as fin:
            for raw in fin:
                line = raw.rstrip('\r\n')
                if line.strip() == "":
                    continue
                lines_out.append(line)

    out_str = "\n".join(lines_out) + ("\n" if lines_out else "")
    with open(mem_data_file_out, 'w') as fout:
        fout.write(out_str)

    return out_str
    