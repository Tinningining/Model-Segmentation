"""
工具调度器
"""


class Device0PreferredScheduler:
    """优先使用Device 0的调度器"""
    
    def __init__(self, devices: list, main_device_id: int = 0):
        self.devices = devices
        self.main_device_id = main_device_id
        self.device_loads = {i: 0 for i in devices}
        self.loaded_tools = {i: set() for i in devices}
    
    def schedule(self, tool_name: str, tool_size: int = 50) -> int:
        """
        调度策略：
        1. 如果工具已在某设备加载，使用该设备
        2. 如果Device 0负载不高，优先使用Device 0
        3. 否则选择负载最低的设备
        """
        # 1. 检查工具是否已在某设备加载
        for device_id, tools in self.loaded_tools.items():
            if tool_name in tools:
                print(f"[Scheduler] Tool '{tool_name}' already loaded on Device {device_id}")
                return device_id
        
        # 2. 检查Device 0的负载
        main_device_load = self.device_loads[self.main_device_id]
        max_load_threshold = 500  # MB
        
        if main_device_load + tool_size < max_load_threshold:
            # Device 0负载不高，优先使用
            print(f"[Scheduler] Scheduling '{tool_name}' to Device {self.main_device_id} (avoid data transfer)")
            self.device_loads[self.main_device_id] += tool_size
            self.loaded_tools[self.main_device_id].add(tool_name)
            return self.main_device_id
        
        # 3. Device 0负载过高，选择其他设备
        other_devices = [i for i in self.devices if i != self.main_device_id]
        if not other_devices:
            # 只有一个设备，强制使用
            print(f"[Scheduler] Only one device available, using Device {self.main_device_id}")
            self.device_loads[self.main_device_id] += tool_size
            self.loaded_tools[self.main_device_id].add(tool_name)
            return self.main_device_id
        
        best_device = min(other_devices, key=lambda x: self.device_loads[x])
        
        print(f"[Scheduler] Scheduling '{tool_name}' to Device {best_device} (Device 0 overloaded)")
        self.device_loads[best_device] += tool_size
        self.loaded_tools[best_device].add(tool_name)
        
        return best_device
    
    def unload_tool(self, tool_name: str, device_id: int, tool_size: int):
        """卸载工具"""
        if device_id in self.loaded_tools and tool_name in self.loaded_tools[device_id]:
            self.loaded_tools[device_id].remove(tool_name)
            self.device_loads[device_id] -= tool_size
            print(f"[Scheduler] Unloaded '{tool_name}' from Device {device_id}")
    
    def get_device_load(self, device_id: int) -> int:
        """获取设备负载"""
        return self.device_loads.get(device_id, 0)
    
    def get_loaded_tools(self, device_id: int) -> set:
        """获取设备上已加载的工具"""
        return self.loaded_tools.get(device_id, set())
