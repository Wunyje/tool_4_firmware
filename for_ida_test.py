import idaapi
import idautils
import idc
import os

print('--------for_ida_test---------')
print(idc.ARGV)
print('--------for_ida_test---------')
processor_name = GetLongPrm(INF_PROCNAME)
print(processor_name)