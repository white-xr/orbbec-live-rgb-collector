# OrbbecLiveCollector 相机采集配置
# ------------------------------------------------------------
# 这个文件只影响 D:\OrbbecLiveCollector\orbbec_live_capture.py 的相机采集。
# 注意：
# 1. 内参、畸变、外参不要手工填在这里；它们由 Orbbec SDK 在当前 pipeline 启动后读取，
#    并保存到每次采集目录里的 camera_info.yaml 和 camera_params.json。
# 2. 曝光、增益、白平衡、激光、深度范围等属于相机设备属性，可以在这里配置。
# 3. 属性值为 None 表示“不修改相机当前值”；只有写成 True/False/整数/浮点数时才会下发到相机。
# 4. 不同型号、固件、USB 模式支持的属性范围可能不同；strict=False 时失败只警告不中断。

CONFIG = {
    # OrbbecSDK.dll 所在目录。你的机器上目前常见是 D:\OrbbecSDK_v2\bin。
    "sdk_bin": r"D:\OrbbecSDK_v2\bin",

    # 采集输出根目录；每次点击 START 会在这里新建 scan_data_时间戳 文件夹。
    "output_root": r"D:\OrbbecLiveCollector\captures",

    # 多相机选择：
    # index 为 None 时默认使用第 0 台；也可以填 0、1、2...
    # serial 非空时优先按序列号选择，适合固定 335L / 305 中某一台。
    "device": {
        "index": None,
        "serial": "",
    },

    "session": {
        # 写入 pose_note.txt / camera_info.yaml 的备注，可填如 base_cover_335L。
        "tag": "",

        # 输出图片 resize。0 表示保持相机对齐后的原始分辨率。
        "output_width": 0,
        "output_height": 0,

        # RGB 与 Depth 时间戳允许的最大差值，超过则跳过该帧对。
        "max_sync_diff_ms": 15.0,
    },

    "stream_profile": {
        # True：对 305 / 335L 使用下面指定的 RGB-D 流规格。
        # False：让 SDK 自动选择默认 RGB/Depth 流。
        "enabled": True,

        # 对 Gemini 305 / Gemini 335L 默认固定到 1280x800@30，保持你之前采集逻辑。
        "use_fixed_profile_for_305_335l": True,

        # 如果指定流规格找不到：
        # False：直接报错，防止采集到不一致的数据；
        # True：退回 SDK 默认流。
        "fallback_to_sdk_default": False,

        # 彩色流。formats 是优先级顺序，脚本会从 SDK 支持列表里挑第一个可用格式。
        "color": {
            "width": 1280,
            "height": 800,
            "fps": 30,
            "formats": ["BGR", "RGB", "YUYV", "MJPG", "BGRA", "RGBA", "UYVY"],
        },

        # 深度流。YOLO + 深度配准建议 Depth 与 RGB 使用可 D2C 对齐的规格。
        "depth": {
            "width": 1280,
            "height": 800,
            "fps": 30,
            "formats": ["Y16"],
        },
    },

    "align": {
        # D2C 对齐顺序：hardware 优先，失败后 software。
        # 如需完全关闭对齐，可改为 ["disable"]，但后续 RGB mask 取深度会不准。
        "mode_order": ["hardware", "software"],

        # 对齐后要求 SDK 返回正确 depth scale，建议保持 True。
        "depth_scale_after_align": True,
    },

    "camera_properties": {
        # 是否启用下面的相机属性写入。
        "enabled": True,

        # False：某个属性不支持时只打印 WARN，继续采集。
        # True：某个属性设置失败就中断启动，适合严谨实验。
        "strict": False,

        "values": {
            # ---------------- RGB 彩色相机曝光/图像参数 ----------------
            # 自动曝光。想手动曝光时先设 False，再设置 COLOR_EXPOSURE / COLOR_GAIN。
            "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL": None,
            # 彩色曝光时间，单位由 Orbbec SDK/固件决定，常见为 us 或设备内部单位。
            "OB_PROP_COLOR_EXPOSURE_INT": None,
            # 彩色增益。
            "OB_PROP_COLOR_GAIN_INT": None,
            # 自动白平衡。想固定颜色时设 False，再设置 COLOR_WHITE_BALANCE。
            "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL": None,
            # 白平衡色温，常见范围如 2800-6500，实际以设备支持为准。
            "OB_PROP_COLOR_WHITE_BALANCE_INT": None,
            "OB_PROP_COLOR_BRIGHTNESS_INT": None,
            "OB_PROP_COLOR_CONTRAST_INT": None,
            "OB_PROP_COLOR_SATURATION_INT": None,
            "OB_PROP_COLOR_SHARPNESS_INT": None,
            "OB_PROP_COLOR_GAMMA_INT": None,
            # 电源频率防闪烁：常见 1=50Hz，2=60Hz，实际以 SDK 文档/设备为准。
            "OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT": None,

            # ---------------- Depth / IR 曝光参数 ----------------
            # Depth/IR 有些型号是同一个物理传感器，设置其中一个可能同步影响另一个。
            "OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL": None,
            "OB_PROP_DEPTH_EXPOSURE_INT": None,
            "OB_PROP_DEPTH_GAIN_INT": None,
            "OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT": None,
            "OB_PROP_IR_AUTO_EXPOSURE_BOOL": None,
            "OB_PROP_IR_EXPOSURE_INT": None,
            "OB_PROP_IR_GAIN_INT": None,

            # ---------------- 深度质量/范围/滤波 ----------------
            # 最小/最大深度阈值，单位通常是 mm。
            "OB_PROP_MIN_DEPTH_INT": None,
            "OB_PROP_MAX_DEPTH_INT": None,
            "OB_PROP_DEPTH_POSTFILTER_BOOL": None,
            "OB_PROP_DEPTH_HOLEFILTER_BOOL": None,
            "OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL": None,
            "OB_PROP_DEPTH_PRECISION_LEVEL_INT": None,

            # ---------------- 激光/补光 ----------------
            "OB_PROP_LASER_BOOL": None,
            # 激光功率等级，实际范围以设备为准。
            "OB_PROP_LASER_POWER_LEVEL_CONTROL_INT": None,
            # 激光电流，单位 mA，谨慎修改。
            "OB_PROP_LASER_CURRENT_FLOAT": None,
            "OB_PROP_FLOOD_BOOL": None,
            "OB_PROP_FLOOD_LEVEL_INT": None,

            # ---------------- 镜像/翻转 ----------------
            # 如果采集图像左右反了，可以在这里设置；注意 RGB/Depth 要保持一致。
            "OB_PROP_COLOR_MIRROR_BOOL": None,
            "OB_PROP_COLOR_FLIP_BOOL": None,
            "OB_PROP_DEPTH_MIRROR_BOOL": None,
            "OB_PROP_DEPTH_FLIP_BOOL": None,
        },
    },
}
