#!/usr/bin/env python
# encoding:utf-8


"""
注册python数据到windows注册表中,用于安装其他Python第三方库
author    :   @h-j-13
time      :   2018-7-21
ref       :   https://www.cnblogs.com/tlz888/p/6879227.html
"""
import sys

from _winreg import *

# tweak as necessary
version = sys.version[:3]
installpath = sys.prefix
regpath = "SOFTWARE\\Python\\Pythoncore\\%s\\" % (version)
installkey = "InstallPath"
pythonkey = "PythonPath"
pythonpath = "%s;%s\\Lib\\;%s\\DLLs\\" % (
    installpath, installpath, installpath
)


def RegisterPy():
    print "begin RegisterPy "
    try:
        print "open key : %s" % regpath
        reg = OpenKey(HKEY_CURRENT_USER, regpath)
    except EnvironmentError as e:
        try:
            reg = CreateKey(HKEY_CURRENT_USER, regpath)
            SetValue(reg, installkey, REG_SZ, installpath)
            SetValue(reg, pythonkey, REG_SZ, pythonpath)
            CloseKey(reg)
        except:
            print "*** EXCEPT: Unable to register!"
            return

        print "--- Python", version, "is now registered!"
        return

    if (QueryValue(reg, installkey) == installpath and
            QueryValue(reg, pythonkey) == pythonpath):
        CloseKey(reg)
        print "=== Python", version, "is already registered!"
        return CloseKey(reg)

    print "*** ERROR:Unable to register!"
    print "*** REASON:You probably have another Python installation!"


def UnRegisterPy():
    # print "begin UnRegisterPy "
    try:
        print "open HKEY_CURRENT_USER key=%s" % (regpath)
        reg = OpenKey(HKEY_CURRENT_USER, regpath)
        # reg = OpenKey(HKEY_LOCAL_MACHINE, regpath)
    except EnvironmentError:
        print "*** Python not registered?!"
        return
    try:
        DeleteKey(reg, installkey)
        DeleteKey(reg, pythonkey)
        DeleteKey(HKEY_LOCAL_MACHINE, regpath)
    except:
        print "*** Unable to un-register!"
    else:
        print "--- Python", version, "is no longer registered!"


if __name__ == "__main__":
    RegisterPy()