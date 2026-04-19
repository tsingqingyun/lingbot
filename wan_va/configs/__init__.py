# # Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# from .va_franka_cfg import va_franka_cfg
# from .va_robotwin_cfg import va_robotwin_cfg
# from .va_franka_i2va import va_franka_i2va_cfg
# from .va_robotwin_i2va import va_robotwin_i2va_cfg
# from .va_robotwin_train_cfg import va_robotwin_train_cfg
# from .va_demo_train_cfg import va_demo_train_cfg
# from .va_demo_cfg import va_demo_cfg
# from .va_demo_i2va import va_demo_i2va_cfg
# from .va_robocasa_cfg import va_robocasa_cfg
# from .va_robocasa_train_cfg import va_robocasa_train_cfg

# VA_CONFIGS = {
#     'robotwin': va_robotwin_cfg,
#     'franka': va_franka_cfg,
#     'robotwin_i2av': va_robotwin_i2va_cfg,
#     'franka_i2av': va_franka_i2va_cfg,
#     'robotwin_train': va_robotwin_train_cfg,
#     'demo': va_demo_cfg,
#     'demo_train': va_demo_train_cfg,
#     'demo_i2av': va_demo_i2va_cfg,
#     'robocasa': va_robocasa_cfg,
#     'robocasa_train': va_robocasa_train_cfg,
# }
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

def get_config(config_name):
    if config_name == 'robotwin':
        from .va_robotwin_cfg import va_robotwin_cfg
        return va_robotwin_cfg
    elif config_name == 'franka':
        from .va_franka_cfg import va_franka_cfg
        return va_franka_cfg
    elif config_name == 'robotwin_i2av':
        from .va_robotwin_i2va import va_robotwin_i2va_cfg
        return va_robotwin_i2va_cfg
    elif config_name == 'franka_i2av':
        from .va_franka_i2va import va_franka_i2va_cfg
        return va_franka_i2va_cfg
    elif config_name == 'robotwin_train':
        from .va_robotwin_train_cfg import va_robotwin_train_cfg
        return va_robotwin_train_cfg
    elif config_name == 'demo':
        from .va_demo_cfg import va_demo_cfg
        return va_demo_cfg
    elif config_name == 'demo_train':
        from .va_demo_train_cfg import va_demo_train_cfg
        return va_demo_train_cfg
    elif config_name == 'demo_i2av':
        from .va_demo_i2va import va_demo_i2va_cfg
        return va_demo_i2va_cfg
    elif config_name == 'robocasa':
        from .va_robocasa_cfg import va_robocasa_cfg
        return va_robocasa_cfg
    elif config_name == 'robocasa_train':
        from .va_robocasa_train_cfg import va_robocasa_train_cfg
        return va_robocasa_train_cfg
    elif config_name == 'libero':
        from .va_libero_cfg import va_libero_cfg
        return va_libero_cfg
    elif config_name == 'libero_train':
        from .va_libero_train_cfg import va_libero_train_cfg
        return va_libero_train_cfg
    elif config_name == 'libero_i2av':
        from .va_libero_i2va import va_libero_i2va_cfg
        return va_libero_i2va_cfg
    else:
        raise KeyError(f"Unknown config name: {config_name}")
