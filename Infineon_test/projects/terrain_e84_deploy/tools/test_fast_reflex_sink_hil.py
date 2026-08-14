import sys, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import fast_reflex_sink_hil as h
def test_quantization_and_saturation():
    raw=np.vstack((h.MEAN,h.MEAN+h.STD*h.SCALE*10000)).astype(np.float32); q=h.quantize(raw)
    assert q[0].tolist()==[-8]*10 and q[1].tolist()==[127]*10
def test_frame_crc_and_order():
    x=np.arange(200,dtype=np.float32).reshape(20,10); f=h.build_frame(7,x); length=int.from_bytes(f[4:6],'little'); assert f[:4]==b'FRV2' and length==806
    payload=f[6:-4]; assert zlib.crc32(payload)&0xffffffff==int.from_bytes(f[-4:],'little'); assert np.array_equal(np.frombuffer(payload[6:],'<f4').reshape(20,10),x)
def test_final_source_guard():
    try: h.guard(Path('/tmp/final_test_9110'))
    except ValueError: return
    assert False
