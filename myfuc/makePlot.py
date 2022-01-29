# import numpy as np
# import struct

# '''
# READ DATA
# '''

# class readWTData(object):
        
#     def makeImg(self,*arg):
#         result=[]
#         for path in arg:
#             fI = open('{}'.format(path), 'rb')
#             nxI = fI.read(4)
#             nxI = struct.unpack("i", nxI)[0]
#             nyI = fI.read(4)
#             nyI = struct.unpack("i", nyI)[0]
#             img_target = []
#             for _ in range(nxI*nyI):
#                 data = fI.read(8)
#                 data = struct.unpack("d", data)[0]
#                 img_target.append(data)
#             fI.close()
#             result.append(np.array(img_target).reshape(nyI, nxI))
#         return result

#     def idwt_h(self, l, h):
#         out=np.empty((l.shape[0],l.shape[1]*2))
#         out[:,::2]=l+h
#         out[:,1::2]=l-h
#         return out * np.sqrt(2)/2
