from myhdl import bin, intbv
from typing import Any, Dict, List, Optional, Tuple
from core.ir.operation import Operation
from core.ir.data import Data, DataType, MemBlock, ViewData, elements_to_32b_cell
from core.ir.prims.lif import PrimLif
import torch
import math
from core.ir.prims.lif import PrimLif

class Lif(Operation):

    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, attrs)
        required_attrs = ['Vmp_rest', 'Vmp_rst', 'Vmp_att', 'Vthr0', 'Vtheta_rst', 
                          'Vtheta_incre', 'Vmp_low', 'Vthr_low', 'A_leaky', 'B_leaky', 'A_theta', 'B_theta', 
                          'Tw_en', 'Tw_len', 'Tw_cnt', 'rst_mode', 'output_mode']
        for attr in required_attrs:
            if attr not in self.attrs:
                raise ValueError(f"Missing required attribute '{attr}' for Lif operation.")
        self.primitive = True  # Marking as a non-primitive operation
    
    
    def infer(self, inputs: Dict[int, Data]) -> List[Tuple[Optional[Tuple[int, ...]], Optional[DataType]]]:
        if len(inputs) != 3:
            raise ValueError("Lif operation requires exactly 3 input.")

        Vin = inputs[0]
        # if Vin.shape is None or Vin.ndim != 2:
        #     raise ValueError("Vin data must be two-dimensional.")
        # assert Vin.dtype == DataType.BF16, "Vin dtype must be bfloat16 format."
        
        Vmp = inputs[1]
        # if Vmp.shape is None or Vmp.ndim != 2:
        #     raise ValueError("Vmp data must be two-dimensional.")
        # assert Vmp.dtype == DataType.BF16, "Vmp dtype must be bfloat16 format."
        
        Vtheta = inputs[2]
        # if Vtheta.shape is None or Vtheta.ndim != 2:
        #     raise ValueError("Vtheta data must be two-dimensional.")
        # assert Vtheta.dtype == DataType.BF16, "Vtheta dtype must be bfloat16 format."

        assert Vin.shape == Vmp.shape == Vtheta.shape, "Vin, Vmp, Vtheta shape do not match."
        if ((self.attrs['output_mode']) == 0 or (self.attrs['output_mode'] == 1)):
            Sout_datatype = DataType.BF16
        elif (self.attrs['output_mode'] == 2):
            Sout_datatype = DataType.SPIKE
        elif (self.attrs['output_mode'] == 3):
            Sout_datatype = DataType.INT8
        Sout_shape = Vin.shape
        Vmp_update_shape = Vin.shape
        Vtheta_update_shape = Vin.shape
        return [(Sout_shape, Sout_datatype), (Vmp_update_shape, DataType.BF16), (Vtheta_update_shape, DataType.BF16)]
    
    
    def para_node(self, inputs: Dict[int, Data], outputs: Dict[int, Data]) -> Optional[Data]:
        """Create a parameter node for lif."""
        Vin = inputs[0]
        Vmp = inputs[1]
        Vtheta = inputs[2]
        Sout = outputs[0]
        Vmp_update = outputs[1]
        Vtheta = outputs[2]
        
        primLif = PrimLif(
            deps=0b00000000, # Placeholder for dependencies
            # Vin=Vin.payload, 
            # Vmp=Vmp.payload,
            # Vtheta=Vtheta.payload,
            # Vin_addr=Vin.memref.addr,
            Vin_addr=16,
            Vmp_addr=16,
            Vtheta_addr=16,
            Sout_addr=16,
            Vmp_update_addr=16,
            para_addr=0, # Placeholder
            Vmp_rest=self.attrs['Vmp_rest'],
            Vmp_rst=self.attrs['Vmp_rst'],
            Vmp_att=self.attrs['Vmp_att'],
            Vthr0=self.attrs['Vthr0'],
            Vtheta_rst=self.attrs['Vtheta_rst'],
            Vtheta_incre=self.attrs['Vtheta_incre'],
            Vmp_low=self.attrs['Vmp_low'],
            Vthr_low=self.attrs['Vthr_low'],
            A_leaky=self.attrs['A_leaky'],
            B_leaky=self.attrs['B_leaky'],
            A_theta=self.attrs['A_theta'],
            B_theta=self.attrs['B_theta'],
            Tw_en=self.attrs['Tw_en'],
            Tw_len=self.attrs['Tw_len'],
            Tw_cnt=self.attrs['Tw_cnt'],
            rst_mode=self.attrs['rst_mode'],
            output_mode=self.attrs['output_mode'],
            input_num=Vin.shape[0],
            input_len_cell=math.ceil(Vin.shape[1]/16)
        )
        
        # TODO: Implement parameter node creation logic
        para_code = MemBlock(length=1, payload=[primLif.para])
        para_data = Data(name=f"{self.name}.params", memref=para_code)
        return para_data
    
    def para_connection(self) -> bool:
        '''
        True: double connection (input and output)
        False: single connection (only input)
        '''
        return True
    
    def gen_prim(self, inputs: Dict[int, Data], outputs: Dict[int, Data], deps=0b00000000) -> intbv:
        Vin = inputs[0]
        Vmp = inputs[1]
        Vtheta = inputs[2]
        para_data_in = inputs[3]
        Sout = outputs[0]
        Vmp_update = outputs[1]
        Vtheta_update = outputs[2]
        para_data_out = outputs[3]
        vin_addr = Vin.inferred_memref.addr if isinstance(Vin, ViewData) else Vin.memref.addr
        
        primLif = PrimLif(
            deps=deps,
            # Vin=Vin.payload, 
            # Vmp=Vmp.payload,
            # Vtheta=Vtheta.payload,
            # Vin_addr=Vin.memref.addr,
            Vin_addr=vin_addr,
            Vmp_addr=Vmp.memref.addr,
            Vtheta_addr=Vtheta.memref.addr,
            Sout_addr=Sout.memref.addr,
            Vmp_update_addr=Vmp_update.memref.addr,
            para_addr=para_data_in.memref.addr, # Placeholder
            Vmp_rest=self.attrs['Vmp_rest'],
            Vmp_rst=self.attrs['Vmp_rst'],
            Vmp_att=self.attrs['Vmp_att'],
            Vthr0=self.attrs['Vthr0'],
            Vtheta_rst=self.attrs['Vtheta_rst'],
            Vtheta_incre=self.attrs['Vtheta_incre'],
            Vmp_low=self.attrs['Vmp_low'],
            Vthr_low=self.attrs['Vthr_low'],
            A_leaky=self.attrs['A_leaky'],
            B_leaky=self.attrs['B_leaky'],
            A_theta=self.attrs['A_theta'],
            B_theta=self.attrs['B_theta'],
            Tw_en=self.attrs['Tw_en'],
            Tw_len=self.attrs['Tw_len'],
            Tw_cnt=self.attrs['Tw_cnt'],
            rst_mode=self.attrs['rst_mode'],
            output_mode=self.attrs['output_mode'],
            input_num=Vin.shape[0],
            input_len_cell=math.ceil(Vin.shape[1]/16)
        )
        
        return primLif.PIC
    
    def build_prim(self,
        inputs: Dict[int, Data],
        outputs: Dict[int, Data],
        deps: intbv = intbv(0)[8:],
    ):
        Vin = inputs[0]
        Vmp = inputs[1]
        Vtheta = inputs[2]
        # 这里输入输出是否应该加上para_data？
        para_data_in = inputs[3]
        Sout = outputs[0]
        Vmp_update = outputs[1]
        Vtheta_update = outputs[2]
        para_data_out = outputs[3]
        primitive = self._build_primitive(Vin, Vmp, Vtheta, para_data_in, Sout, Vmp_update, Vtheta_update, para_data_out, deps)
        # primitive = self._build_primitive(Vin, Vmp, Vtheta, Sout, Vmp_update, Vtheta_update, deps)
        return primitive

    def _build_primitive(
        self,
        Vin: Data,
        Vmp: Data,
        Vtheta: Data,
        # 不确定是不是Data类型
        para_data_in: Data,
        Sout: Data,
        Vmp_update: Data,
        Vtheta_update: Data,
        # 不确定是不是Data类型
        para_data_out: Data,
        deps: intbv,
    ) -> PrimLif:

        rst_mode = self.attrs["rst_mode"]
        output_mode = self.attrs["output_mode"]

        # if rst_mode not in (0, 1, 2, 3):
        #     assert False, "Vmp rst mode must be 0, 1, 2 or 3."
        # if output_mode not in (0, 1, 2, 3):
        #     assert False, "Sout output mode must be 0, 1, 2 or 3."

        # 取tensor_size除最后一维的乘积作为input_num
        input_num = math.prod(Vin.shape[:-1])
        # 取最后一维转换为32B单元的长度作为input_len_cell(占几个cell)
        input_len_cell = elements_to_32b_cell(Vin.shape[-1], Vin.dtype)
        # vin_addr = Vin.inferred_memref.addr if isinstance(Vin, ViewData) else Vin.memref.addr

        return PrimLif(
            deps=int(deps),
            Vin_addr=Vin.memref.addr,
            Vmp_addr=Vmp.memref.addr,
            Vtheta_addr=Vtheta.memref.addr,
            Sout_addr=Sout.memref.addr,
            Vmp_update_addr=Vmp_update.memref.addr,
            para_addr=para_data_in.memref.addr, # Placeholder
            Vmp_rest=self.attrs['Vmp_rest'],
            Vmp_rst=self.attrs['Vmp_rst'],
            Vmp_att=self.attrs['Vmp_att'],
            Vthr0=self.attrs['Vthr0'],
            Vtheta_rst=self.attrs['Vtheta_rst'],
            Vtheta_incre=self.attrs['Vtheta_incre'],
            Vmp_low=self.attrs['Vmp_low'],
            Vthr_low=self.attrs['Vthr_low'],
            A_leaky=self.attrs['A_leaky'],
            B_leaky=self.attrs['B_leaky'],
            A_theta=self.attrs['A_theta'],
            B_theta=self.attrs['B_theta'],
            Tw_en=self.attrs['Tw_en'],
            Tw_len=self.attrs['Tw_len'],
            Tw_cnt=self.attrs['Tw_cnt'],
            rst_mode=self.attrs['rst_mode'],
            output_mode=self.attrs['output_mode'],
            input_num=input_num,
            input_len_cell=input_len_cell,
        )
        