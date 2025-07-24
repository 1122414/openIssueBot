import sys
import os
import site
import pkg_resources

print(f'Python版本: {sys.version}')
print(f'Python路径: {sys.executable}')
print(f'Python安装目录: {sys.prefix}')

# 检查是否在虚拟环境中
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print(f'当前在虚拟环境中: {sys.prefix}')
else:
    print('当前不在虚拟环境中')

# 显示系统路径
print('\nPython路径:')
for p in sys.path:
    print(f'  - {p}')

# 显示环境变量
print('\n环境变量:')
for key, value in os.environ.items():
    if 'PYTHON' in key or 'PATH' in key or 'VIRTUAL_ENV' in key:
        print(f'{key}: {value}')

print('\n已安装的包:')
for d in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower()):
    print(f"{d.project_name}=={d.version}")