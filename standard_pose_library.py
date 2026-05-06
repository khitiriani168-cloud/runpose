# encoding: utf-8
"""
标准跑姿库 - 包含专业认证的标准跑姿数据
数据结构：跑步类型 → 性别 → 视角 → 参数曲线
"""
import numpy as np

# ==================== 标准跑姿库数据 ====================
STANDARD_POSE_LIBRARY = {
    "sprint": {  # 短跑
        "male": {
            "side_view": {
                "description": "男性短跑标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 155,      # 着地期膝角
                        "stance": 145,       # 支撑期膝角
                        "push_off": 165,     # 蹬伸期膝角
                        "swing": 45          # 摆动期膝角（折叠）
                    },
                    "hip_extension": {
                        "contact": 150,
                        "stance": 152,
                        "push_off": 172,
                        "swing": 125
                    },
                    "torso_lean": {
                        "contact": 12,
                        "stance": 11,
                        "push_off": 13,
                        "swing": 10
                    },
                    "elbow_flexion": {
                        "contact": 95,
                        "stance": 90,
                        "push_off": 100,
                        "swing": 92
                    },
                    "stride_ratio": 2.5,
                    "vertical_amplitude": 12,
                    "ideal_ranges": {
                        "knee_flexion_min": 35,
                        "knee_flexion_max": 175,
                        "hip_extension_min": 120,
                        "hip_extension_max": 175,
                        "torso_lean_min": 8,
                        "torso_lean_max": 18,
                        "stride_ratio_min": 2.0,
                        "stride_ratio_max": 3.0
                    }
                },
                "normalized_cycles": {
                    "knee_flexion": [155, 150, 145, 142, 148, 155, 162, 165, 160, 145, 120, 95, 70, 55, 45, 50, 65, 85, 110, 135, 150, 155],
                    "hip_extension": [150, 151, 152, 153, 155, 158, 162, 168, 172, 170, 165, 158, 148, 138, 130, 128, 132, 138, 145, 148, 149, 150],
                    "torso_lean": [12, 12, 11, 11, 11, 11, 12, 13, 13, 12, 11, 10, 10, 10, 10, 10, 10, 11, 11, 12, 12, 12],
                    "elbow_flexion": [95, 93, 90, 88, 88, 89, 92, 95, 100, 98, 95, 93, 91, 90, 90, 91, 92, 93, 94, 94, 95, 95]
                }
            },
            "front_view": {
                "description": "男性短跑标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -2,
                        "push_off": 0,
                        "swing": 3
                    },
                    "shoulder_rotation": {
                        "contact": 15,
                        "stance": 12,
                        "push_off": 18,
                        "swing": 14
                    },
                    "arm_symmetry": 0.95,
                    "leg_symmetry": 0.94,
                    "ideal_ranges": {
                        "knee_adduction_min": -5,
                        "knee_adduction_max": 5,
                        "arm_symmetry_min": 0.9,
                        "leg_symmetry_min": 0.9
                    }
                }
            }
        },
        "female": {
            "side_view": {
                "description": "女性短跑标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 158,
                        "stance": 148,
                        "push_off": 168,
                        "swing": 48
                    },
                    "hip_extension": {
                        "contact": 152,
                        "stance": 154,
                        "push_off": 170,
                        "swing": 128
                    },
                    "torso_lean": {
                        "contact": 10,
                        "stance": 9,
                        "push_off": 11,
                        "swing": 8
                    },
                    "elbow_flexion": {
                        "contact": 92,
                        "stance": 88,
                        "push_off": 98,
                        "swing": 90
                    },
                    "stride_ratio": 2.3,
                    "vertical_amplitude": 10,
                    "ideal_ranges": {
                        "knee_flexion_min": 40,
                        "knee_flexion_max": 175,
                        "hip_extension_min": 125,
                        "hip_extension_max": 172,
                        "torso_lean_min": 6,
                        "torso_lean_max": 15,
                        "stride_ratio_min": 1.8,
                        "stride_ratio_max": 2.8
                    }
                }
            },
            "front_view": {
                "description": "女性短跑标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -3,
                        "push_off": 0,
                        "swing": 2
                    },
                    "shoulder_rotation": {
                        "contact": 12,
                        "stance": 10,
                        "push_off": 15,
                        "swing": 11
                    },
                    "arm_symmetry": 0.96,
                    "leg_symmetry": 0.95,
                    "ideal_ranges": {
                        "knee_adduction_min": -6,
                        "knee_adduction_max": 4,
                        "arm_symmetry_min": 0.92,
                        "leg_symmetry_min": 0.92
                    }
                }
            }
        }
    },
    "middle_distance": {  # 中长跑
        "male": {
            "side_view": {
                "description": "男性中长跑标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 160,      # 着地瞬间约160度
                        "stance": 140,       # 支撑期逐渐屈曲到约140度
                        "push_off": 168,     # 蹬伸期伸展到约168度
                        "swing": 55          # 摆动期迅速折叠到约55度
                    },
                    "hip_extension": {
                        "contact": 155,      # 着地瞬间约155度
                        "stance": 155,       # 支撑期保持约155度
                        "push_off": 170,     # 蹬伸期伸展到约170度
                        "swing": 130         # 摆动期屈曲到约130度
                    },
                    "torso_lean": {
                        "contact": 6,        # 躯干前倾角维持在5-8度
                        "stance": 5,
                        "push_off": 7,
                        "swing": 6
                    },
                    "elbow_flexion": {
                        "contact": 90,       # 着地期约90度
                        "stance": 85,        # 支撑期约85度
                        "push_off": 95,      # 蹬伸期约95度
                        "swing": 90          # 摆动期约90度
                    },
                    "stride_ratio": 1.5,
                    "vertical_amplitude": 8,
                    "ideal_ranges": {
                        "knee_flexion_min": 50,
                        "knee_flexion_max": 170,
                        "hip_extension_min": 130,
                        "hip_extension_max": 172,
                        "torso_lean_min": 3,
                        "torso_lean_max": 12,
                        "stride_ratio_min": 1.3,
                        "stride_ratio_max": 1.8
                    }
                },
                "normalized_cycles": {
                    "knee_flexion": [160, 158, 152, 145, 140, 142, 148, 155, 162, 168, 165, 158, 145, 125, 100, 75, 55, 60, 80, 105, 135, 155, 160],
                    "hip_extension": [155, 155, 155, 156, 157, 158, 160, 163, 167, 170, 168, 165, 160, 152, 142, 135, 130, 132, 138, 145, 152, 154, 155],
                    "torso_lean": [6, 6, 5, 5, 5, 5, 6, 6, 7, 7, 6, 6, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
                    "elbow_flexion": [90, 88, 85, 85, 86, 87, 88, 90, 92, 95, 93, 91, 90, 89, 90, 90, 90, 91, 92, 92, 91, 90, 90]
                }
            },
            "front_view": {
                "description": "男性中长跑标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -1,
                        "push_off": 0,
                        "swing": 2
                    },
                    "shoulder_rotation": {
                        "contact": 10,
                        "stance": 8,
                        "push_off": 12,
                        "swing": 9
                    },
                    "arm_symmetry": 0.97,
                    "leg_symmetry": 0.96,
                    "ideal_ranges": {
                        "knee_adduction_min": -4,
                        "knee_adduction_max": 4,
                        "arm_symmetry_min": 0.93,
                        "leg_symmetry_min": 0.93
                    }
                }
            }
        },
        "female": {
            "side_view": {
                "description": "女性中长跑标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 162,
                        "stance": 142,
                        "push_off": 168,
                        "swing": 58
                    },
                    "hip_extension": {
                        "contact": 155,
                        "stance": 155,
                        "push_off": 168,
                        "swing": 132
                    },
                    "torso_lean": {
                        "contact": 5,
                        "stance": 4,
                        "push_off": 6,
                        "swing": 5
                    },
                    "elbow_flexion": {
                        "contact": 88,
                        "stance": 84,
                        "push_off": 92,
                        "swing": 88
                    },
                    "stride_ratio": 1.4,
                    "vertical_amplitude": 7,
                    "ideal_ranges": {
                        "knee_flexion_min": 55,
                        "knee_flexion_max": 170,
                        "hip_extension_min": 132,
                        "hip_extension_max": 170,
                        "torso_lean_min": 2,
                        "torso_lean_max": 10,
                        "stride_ratio_min": 1.2,
                        "stride_ratio_max": 1.7
                    }
                }
            },
            "front_view": {
                "description": "女性中长跑标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -2,
                        "push_off": 0,
                        "swing": 1
                    },
                    "shoulder_rotation": {
                        "contact": 8,
                        "stance": 6,
                        "push_off": 10,
                        "swing": 7
                    },
                    "arm_symmetry": 0.98,
                    "leg_symmetry": 0.97,
                    "ideal_ranges": {
                        "knee_adduction_min": -5,
                        "knee_adduction_max": 3,
                        "arm_symmetry_min": 0.95,
                        "leg_symmetry_min": 0.95
                    }
                }
            }
        }
    },
    "jogging": {  # 慢跑
        "male": {
            "side_view": {
                "description": "男性慢跑标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 165,
                        "stance": 145,
                        "push_off": 165,
                        "swing": 65
                    },
                    "hip_extension": {
                        "contact": 152,
                        "stance": 152,
                        "push_off": 165,
                        "swing": 135
                    },
                    "torso_lean": {
                        "contact": 4,
                        "stance": 3,
                        "push_off": 5,
                        "swing": 4
                    },
                    "elbow_flexion": {
                        "contact": 88,
                        "stance": 85,
                        "push_off": 92,
                        "swing": 88
                    },
                    "stride_ratio": 1.2,
                    "vertical_amplitude": 6,
                    "ideal_ranges": {
                        "knee_flexion_min": 60,
                        "knee_flexion_max": 170,
                        "hip_extension_min": 135,
                        "hip_extension_max": 170,
                        "torso_lean_min": 1,
                        "torso_lean_max": 10,
                        "stride_ratio_min": 1.1,
                        "stride_ratio_max": 1.4
                    }
                }
            },
            "front_view": {
                "description": "男性慢跑标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -1,
                        "push_off": 0,
                        "swing": 1
                    },
                    "shoulder_rotation": {
                        "contact": 8,
                        "stance": 6,
                        "push_off": 10,
                        "swing": 7
                    },
                    "arm_symmetry": 0.98,
                    "leg_symmetry": 0.97,
                    "ideal_ranges": {
                        "knee_adduction_min": -3,
                        "knee_adduction_max": 3,
                        "arm_symmetry_min": 0.95,
                        "leg_symmetry_min": 0.95
                    }
                }
            }
        },
        "female": {
            "side_view": {
                "description": "女性慢跑标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 168,
                        "stance": 148,
                        "push_off": 168,
                        "swing": 68
                    },
                    "hip_extension": {
                        "contact": 152,
                        "stance": 152,
                        "push_off": 165,
                        "swing": 138
                    },
                    "torso_lean": {
                        "contact": 3,
                        "stance": 2,
                        "push_off": 4,
                        "swing": 3
                    },
                    "elbow_flexion": {
                        "contact": 86,
                        "stance": 83,
                        "push_off": 90,
                        "swing": 86
                    },
                    "stride_ratio": 1.15,
                    "vertical_amplitude": 5,
                    "ideal_ranges": {
                        "knee_flexion_min": 65,
                        "knee_flexion_max": 172,
                        "hip_extension_min": 138,
                        "hip_extension_max": 168,
                        "torso_lean_min": 0,
                        "torso_lean_max": 8,
                        "stride_ratio_min": 1.0,
                        "stride_ratio_max": 1.3
                    }
                }
            },
            "front_view": {
                "description": "女性慢跑标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -1,
                        "push_off": 0,
                        "swing": 1
                    },
                    "shoulder_rotation": {
                        "contact": 6,
                        "stance": 5,
                        "push_off": 8,
                        "swing": 6
                    },
                    "arm_symmetry": 0.98,
                    "leg_symmetry": 0.98,
                    "ideal_ranges": {
                        "knee_adduction_min": -4,
                        "knee_adduction_max": 2,
                        "arm_symmetry_min": 0.96,
                        "leg_symmetry_min": 0.96
                    }
                }
            }
        }
    },
    "walking": {  # 竞走
        "male": {
            "side_view": {
                "description": "男性竞走标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 165,      # 着地时接近伸直
                        "stance": 160,
                        "push_off": 170,
                        "swing": 155
                    },
                    "hip_extension": {
                        "contact": 145,
                        "stance": 148,
                        "push_off": 160,
                        "swing": 135
                    },
                    "torso_lean": {
                        "contact": 2,
                        "stance": 1,
                        "push_off": 3,
                        "swing": 2
                    },
                    "elbow_flexion": {
                        "contact": 90,
                        "stance": 88,
                        "push_off": 92,
                        "swing": 90
                    },
                    "stride_ratio": 1.0,
                    "vertical_amplitude": 3,
                    "ideal_ranges": {
                        "knee_flexion_min": 150,
                        "knee_flexion_max": 175,
                        "hip_extension_min": 140,
                        "hip_extension_max": 165,
                        "torso_lean_min": 0,
                        "torso_lean_max": 5,
                        "stride_ratio_min": 0.9,
                        "stride_ratio_max": 1.2
                    }
                }
            },
            "front_view": {
                "description": "男性竞走标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -1,
                        "push_off": 0,
                        "swing": 1
                    },
                    "shoulder_rotation": {
                        "contact": 5,
                        "stance": 4,
                        "push_off": 6,
                        "swing": 5
                    },
                    "arm_symmetry": 0.99,
                    "leg_symmetry": 0.98,
                    "ideal_ranges": {
                        "knee_adduction_min": -3,
                        "knee_adduction_max": 3,
                        "arm_symmetry_min": 0.97,
                        "leg_symmetry_min": 0.96
                    }
                }
            }
        },
        "female": {
            "side_view": {
                "description": "女性竞走标准跑姿 - 侧面视角",
                "gait_cycle_params": {
                    "knee_flexion": {
                        "contact": 168,
                        "stance": 162,
                        "push_off": 172,
                        "swing": 158
                    },
                    "hip_extension": {
                        "contact": 145,
                        "stance": 148,
                        "push_off": 158,
                        "swing": 138
                    },
                    "torso_lean": {
                        "contact": 1,
                        "stance": 0,
                        "push_off": 2,
                        "swing": 1
                    },
                    "elbow_flexion": {
                        "contact": 88,
                        "stance": 86,
                        "push_off": 90,
                        "swing": 88
                    },
                    "stride_ratio": 0.95,
                    "vertical_amplitude": 2.5,
                    "ideal_ranges": {
                        "knee_flexion_min": 155,
                        "knee_flexion_max": 175,
                        "hip_extension_min": 142,
                        "hip_extension_max": 162,
                        "torso_lean_min": 0,
                        "torso_lean_max": 4,
                        "stride_ratio_min": 0.85,
                        "stride_ratio_max": 1.15
                    }
                }
            },
            "front_view": {
                "description": "女性竞走标准跑姿 - 正面视角",
                "gait_cycle_params": {
                    "knee_adduction": {
                        "contact": 0,
                        "stance": -1,
                        "push_off": 0,
                        "swing": 1
                    },
                    "shoulder_rotation": {
                        "contact": 4,
                        "stance": 3,
                        "push_off": 5,
                        "swing": 4
                    },
                    "arm_symmetry": 0.99,
                    "leg_symmetry": 0.99,
                    "ideal_ranges": {
                        "knee_adduction_min": -2,
                        "knee_adduction_max": 2,
                        "arm_symmetry_min": 0.98,
                        "leg_symmetry_min": 0.98
                    }
                }
            }
        }
    }
}

# ==================== 雷达图维度定义 ====================
RADAR_DIMENSIONS = {
    "core_stability": {
        "name": "核心稳定度",
        "description": "躯干在跑步过程中的稳定程度",
        "weight": 0.20,
        "sub_metrics": [
            {"name": "躯干前倾角波动", "key": "torso_lean_variance", "unit": "度"},
            {"name": "躯干侧倾角波动", "key": "torso_side_variance", "unit": "度"},
            {"name": "骨盆倾斜度", "key": "pelvic_tilt", "unit": "%"},
            {"name": "头部晃动", "key": "head_tilt", "unit": "%"}
        ]
    },
    "leg_fold_efficiency": {
        "name": "下肢折叠效率",
        "description": "下肢在摆动期折叠和在蹬伸期伸展的效率",
        "weight": 0.25,
        "sub_metrics": [
            {"name": "膝关节折叠角度", "key": "knee_flex_min", "unit": "度"},
            {"name": "膝关节伸展角度", "key": "knee_flex_max", "unit": "度"},
            {"name": "髋关节伸展角度", "key": "hip_extension", "unit": "度"},
            {"name": "折叠速率", "key": "fold_rate", "unit": "度/帧"}
        ]
    },
    "landing_quality": {
        "name": "着地品质",
        "description": "脚着地时的位置和方式",
        "weight": 0.20,
        "sub_metrics": [
            {"name": "着地点偏差", "key": "landing_offset", "unit": "%身高"},
            {"name": "着地膝角", "key": "landing_knee_angle", "unit": "度"},
            {"name": "踝关节角度", "key": "ankle_angle", "unit": "度"}
        ]
    },
    "propulsion": {
        "name": "推进力",
        "description": "蹬伸阶段的发力质量",
        "weight": 0.18,
        "sub_metrics": [
            {"name": "髋角伸展", "key": "hip_extension_final", "unit": "度"},
            {"name": "踝关节跖屈幅度", "key": "ankle_plantarflexion", "unit": "度"},
            {"name": "蹬伸时间占比", "key": "push_off_ratio", "unit": "%"}
        ]
    },
    "symmetry": {
        "name": "对称性",
        "description": "左右两侧肢体动作的一致程度",
        "weight": 0.17,
        "sub_metrics": [
            {"name": "膝关节角度相似度", "key": "knee_sym_correlation", "unit": ""},
            {"name": "髋关节角度相似度", "key": "hip_sym_correlation", "unit": ""},
            {"name": "触地时间比值", "key": "stance_time_ratio", "unit": ""},
            {"name": "摆臂幅度差异", "key": "arm_swing_diff", "unit": "度"}
        ]
    }
}

# ==================== 优化建议规则库 ====================
OPTIMIZATION_RULES = {
    "core_stability": {
        "severe": {
            "condition": "score < 40",
            "advice": "核心稳定性严重不足，建议暂停高强度训练。每天进行10分钟平板支撑、死虫式和农夫行走练习，重建核心力量基础。",
            "exercises": [
                {
                    "name": "平板支撑",
                    "description": "肘部支撑身体呈直线，核心收紧保持稳定",
                    "standard": "双肘在肩关节正下方撑地，双脚与肩同宽，脚尖着地。收紧腹部和臀部，保持身体从侧面看呈一条笔直直线，不塌腰、不弓背、不抬臀。头颈自然延伸，目光看向地面。全程保持均匀呼吸，不要憋气。",
                    "sets": 3,
                    "duration": "20-30秒/组",
                    "rest": "30-45秒",
                    "tempo": "保持匀速呼吸，每组坚持到规定时间",
                    "frequency": "每天",
                    "target": "强化核心肌群"
                },
                {
                    "name": "死虫式",
                    "description": "仰卧交替伸展对侧手脚，核心维持稳定",
                    "standard": "仰卧在垫上，双臂伸直举向天花板，双膝抬起呈90°，小腿与地面平行。收紧核心将腰部贴实地面（不留空隙）。缓慢同时放下右手和左腿至接近地面但不触地，然后回到起始位，换对侧进行。动作过程中腰部始终贴地不抬起。",
                    "sets": 3,
                    "reps": "每侧8-10次",
                    "rest": "30-45秒",
                    "tempo": "每侧放下过程3秒，收回2秒，控制节奏",
                    "frequency": "每天",
                    "target": "改善核心控制"
                },
                {
                    "name": "农夫行走",
                    "description": "双手持重物行走，保持躯干稳定不晃动",
                    "standard": "双手各持一个哑铃或壶铃（初学者2-5kg），挺胸收肩，核心收紧，肩胛骨下沉后收。行走时保持上半身稳定不左右晃动，步幅自然，背部挺直不旋转。目光平视前方，每步落地轻稳。",
                    "sets": 3,
                    "distance": "15-20米/组",
                    "rest": "45-60秒",
                    "tempo": "匀速行走，每步稳定控制，不追求速度",
                    "frequency": "隔天",
                    "target": "提升核心稳定性"
                }
            ]
        },
        "moderate": {
            "condition": "score >= 40 and score < 60",
            "advice": "核心稳定性需要加强。标准跑姿要求：上半身保持直立微前倾约5-10°，不塌腰、不驼背、不过度后仰，核心收紧，耳朵、肩膀、髋骨在一条直线上。头部稳定，平视前方，不低头不仰头，颈部自然放松。",
            "exercises": [
                {
                    "name": "平板支撑",
                    "description": "肘部支撑身体呈直线，核心收紧保持稳定",
                    "standard": "双肘在肩关节正下方撑地，双脚与肩同宽，脚尖着地。收紧腹部和臀部，保持身体从侧面看呈一条笔直直线，不塌腰、不弓背、不抬臀。头颈自然延伸，目光看向地面。全程保持均匀呼吸，不要憋气。",
                    "sets": 4,
                    "duration": "45-60秒/组",
                    "rest": "30-45秒",
                    "tempo": "匀速呼吸，最后10秒可适当加快呼吸频率",
                    "frequency": "每周3-4次",
                    "target": "强化核心"
                },
                {
                    "name": "鸟狗式",
                    "description": "四点支撑交替伸展对侧手脚，维持骨盆稳定",
                    "standard": "四足跪姿，双膝在髋关节正下方，双手在肩关节正下方。背部保持平坦，核心收紧。缓慢同时向前伸直右手臂、向后伸直左腿，至与身体平行为止，停顿1-2秒。感受臀部和背部发力，然后缓慢收回。换对侧进行。全程骨盆不旋转、不塌腰。",
                    "sets": 3,
                    "reps": "每侧10-12次",
                    "rest": "30-45秒",
                    "tempo": "伸展过程3秒，顶端停留2秒，收回3秒",
                    "frequency": "每周3-4次",
                    "target": "改善平衡"
                },
                {
                    "name": "侧平板",
                    "description": "侧身单肘支撑，身体呈直线",
                    "standard": "侧卧，右肘在右肩正下方撑地，双腿伸直并拢叠放。收紧核心将髋部抬离地面，使身体从头到脚呈一条直线。左手可叉腰或伸直指向天花板。保持均匀呼吸，髋部不塌陷。做完一侧换另一侧。",
                    "sets": 3,
                    "duration": "30-45秒/侧",
                    "rest": "30秒（两侧之间）+45秒（组间）",
                    "tempo": "匀速呼吸，全程核心收紧",
                    "frequency": "每周2-3次",
                    "target": "强化侧核心"
                },
                {
                    "name": "俄罗斯转体",
                    "description": "坐姿旋转躯干，强化腹外斜肌和核心旋转力量",
                    "standard": "坐于垫上，双膝屈曲90°，脚跟着地（进阶可悬空）。上半身后倾约45°，收紧核心保持背部挺直不弓背。双手合十在前方，用核心发力向左旋转躯干至极限，停顿1秒后返回中间，再向右旋转。全程保持腹部收紧，骨盆稳定不晃动。可手持哑铃/药球增加负重。",
                    "sets": 3,
                    "reps": "每侧12-15次",
                    "rest": "45-60秒",
                    "tempo": "旋转2秒，顶端停顿1秒，返回2秒",
                    "frequency": "每周2次",
                    "target": "强化核心旋转力量"
                }
            ]
        },
        "mild": {
            "condition": "score >= 60 and score < 80",
            "advice": "核心稳定性良好。注意保持上半身直立微前倾5-10°，核心收紧，耳朵-肩膀-髋骨三点一线，头部平视前方，颈部放松。",
            "exercises": [
                {
                    "name": "悬垂举腿",
                    "description": "单杠悬垂，双腿伸直上举至与地面平行",
                    "standard": "双手正握单杠略宽于肩，身体自然悬垂。收紧核心，保持身体不晃动。缓慢抬起双腿至与地面平行（或更高），在顶端停顿1-2秒感受下腹发力，然后缓慢放下。全程身体不借摆动惯性，腿部尽量保持伸直。如无法完成直腿，可先做屈膝举腿。",
                    "sets": 3,
                    "reps": "8-12次",
                    "rest": "60-90秒",
                    "tempo": "上举2秒，顶端停顿1-2秒，放下3秒",
                    "frequency": "每周2次",
                    "target": "强化下核心"
                },
                {
                    "name": "绳索伐木",
                    "description": "使用弹力带或器械进行旋转核心发力动作",
                    "standard": "侧身站在龙门架或弹力带固定点旁，双手握住把手。从斜上方向斜下方做砍伐动作（高位到低位），过程中核心收紧、髋部旋转带动上半身。动作结束时双手在另一侧膝盖外侧。控制回放速度，不借惯性。做完一侧换另一侧。",
                    "sets": 3,
                    "reps": "每侧10-12次",
                    "rest": "45-60秒",
                    "tempo": "发力过程2秒，回放3秒，控制节奏",
                    "frequency": "每周2次",
                    "target": "提升核心旋转力量"
                }
            ]
        }
    },
    "leg_fold_efficiency": {
        "severe": {
            "condition": "score < 40",
            "advice": "下肢折叠效率严重不足，这会严重影响跑步效率并增加受伤风险。建议从基础力量训练开始，重点强化腘绳肌和髋屈肌。",
            "exercises": [
                {
                    "name": "北欧弯举",
                    "description": "同伴固定双脚，身体前倾后恢复，主要强化腘绳肌",
                    "standard": "双膝跪地，同伴或器械固定脚踝。保持身体从膝盖到肩膀呈一条直线，核心收紧不塌腰。缓慢控制身体前倾，感受大腿后侧被拉伸，直到无法控制时用手撑地缓冲。用手推回起始位置，不要用腘绳肌发力拉回。初学者可减少幅度，只下降30-45°。",
                    "sets": 3,
                    "reps": "6-8次",
                    "rest": "60-90秒",
                    "tempo": "下降过程4-5秒，推回2秒",
                    "frequency": "每周2次",
                    "target": "强化腘绳肌"
                },
                {
                    "name": "高抬腿",
                    "description": "原地高抬腿跑，膝盖抬至腰部高度",
                    "standard": "站立位，挺胸收腹，核心收紧。交替快速向上提膝至大腿与地面平行（或更高），脚尖勾起。落地时前脚掌着地，轻盈有弹性。保持上半身稳定不后仰，双臂配合做摆臂动作。膝盖主动上抬而非靠腿发力蹬地。",
                    "sets": 4,
                    "duration": "20-30秒/组",
                    "rest": "45-60秒",
                    "tempo": "快速提膝，落地即起，保持节奏",
                    "frequency": "每周3次",
                    "target": "提升髋屈肌力量"
                },
                {
                    "name": "弹力带臀桥",
                    "description": "臀部发力顶起身体，弹力带增加阻力",
                    "standard": "仰卧，双膝屈曲90°，双脚与肩同宽踩地。弹力带套在髋部上方（或膝盖上方增加外展刺激）。收紧核心，用臀部发力将髋部向上顶起至身体从肩膀到膝盖呈一条直线。在顶端用力夹紧臀部2秒，然后缓慢放下至接近地面但不完全触地。",
                    "sets": 3,
                    "reps": "15-20次",
                    "rest": "45-60秒",
                    "tempo": "上顶2秒，顶端夹紧2秒，放下3秒",
                    "frequency": "每周3次",
                    "target": "强化臀部"
                }
            ]
        },
        "moderate": {
            "condition": "score >= 40 and score < 60",
            "advice": "下肢折叠效率有待提高。摆动期应充分折叠小腿（脚跟踢向臀部），蹬伸期充分伸展髋关节。支撑腿和摆动腿动作需连贯，避免甩小腿或拖地。建议通过踢臀跑、A-skip等技术训练改善。",
            "exercises": [
                {
                    "name": "踢臀跑",
                    "description": "小步跑，脚跟快速踢向臀部",
                    "standard": "站立位，核心收紧，双臂正常摆臂。小步向前跑动，每步主动将脚跟向上勾起踢向臀部，感受腘绳肌的收缩。膝盖不上抬过高，重点在脚跟勾起的幅度和速度。落地时前脚掌着地，步幅小、频率快。保持上半身微前倾。",
                    "sets": 4,
                    "distance": "20-30米/组",
                    "rest": "45-60秒",
                    "tempo": "快速勾腿，落地即起，保持高频节奏",
                    "frequency": "每周2次",
                    "target": "提高折叠速度"
                },
                {
                    "name": "A-skip",
                    "description": "弹性跳跃，膝盖高抬，前脚掌弹性着地",
                    "standard": "站立位，核心收紧。做一个带有弹跳的提膝动作：左脚向前做一个小跳步，同时右腿提膝至大腿与地面平行，脚尖勾起。落地时前脚掌先着地并迅速弹起切换至另一侧。动作要有弹性和节奏感，膝盖主动上抬而非靠爆发力硬跳。双臂自然摆臂配合。",
                    "sets": 4,
                    "distance": "20-30米/组",
                    "rest": "45-60秒",
                    "tempo": "一左一右为一次，保持弹性节奏",
                    "frequency": "每周2次",
                    "target": "提升下肢弹性"
                },
                {
                    "name": "台阶训练",
                    "description": "快速上下台阶，强化小腿和腘绳肌",
                    "standard": "面对台阶（高度20-30cm），挺胸收腹。左脚踩上台阶，右腿跟上站直，然后右脚先下、左脚跟下。保持节奏连续进行。动作过程中上身保持直立不前倾，核心收紧不晃动。每步踩实后再发力，不半脚掌悬空。",
                    "sets": 3,
                    "reps": "每侧15-20次",
                    "rest": "45-60秒",
                    "tempo": "上步2秒，下步2秒，匀速控制",
                    "frequency": "每周2次",
                    "target": "强化小腿和腘绳肌"
                },
                {
                    "name": "提膝",
                    "description": "站立交替提膝至胸口，强化髋屈肌提升膝盖驱动能力",
                    "standard": "站立位，挺胸收腹，核心收紧。双手叉腰或自然下垂。右腿支撑，左腿主动向上提膝至大腿尽可能靠近胸口（至少大腿与地面平行），脚尖勾起，膝盖主动上抬而非靠身体晃动借力。在最高点停顿1-2秒，感受髋屈肌收缩。然后缓慢放下，换另一侧。全程保持上半身挺直不后仰不前倾，支撑腿微屈膝缓冲。可进阶为站立提膝跳增加爆发力。",
                    "sets": 3,
                    "reps": "每侧15-20次",
                    "rest": "30-45秒",
                    "tempo": "上提2秒，顶端停顿1-2秒，下放2秒",
                    "frequency": "每天",
                    "target": "强化髋屈肌提升膝盖驱动"
                }
            ]
        },
        "mild": {
            "condition": "score >= 60 and score < 80",
            "advice": "下肢折叠效率良好。可以尝试一些进阶练习来进一步提升，如加速跑和阻力跑。",
            "exercises": [
                {
                    "name": "加速跑",
                    "description": "从慢跑逐渐加速到最大速度，提升爆发力",
                    "standard": "选择60-80米平坦跑道。从站立起跑姿势开始，前10米慢跑加速，10-30米逐渐加大步频和步幅，30-50米达全速冲刺，50-60米保持速度，最后10-20米逐渐减速。全程保持躯干直立微前倾，摆臂有力，高抬膝盖，落地轻盈。",
                    "sets": 4,
                    "distance": "60-80米/组",
                    "rest": "90-120秒（走回起点或慢走恢复）",
                    "tempo": "逐渐加速—全速保持—逐渐减速",
                    "frequency": "每周1次",
                    "target": "提升爆发力"
                },
                {
                    "name": "阻力伞跑",
                    "description": "拖着阻力伞跑步，强化推进力",
                    "standard": "将阻力伞固定在腰部后方，确保绳索无缠绕。从站立位启动，前几步注意身体前倾角度略大以克服阻力。跑动时保持高抬膝盖、充分后蹬，步幅适中、频率稳定。阻力伞展开后保持匀速奔跑，不被阻力拉后仰。到达终点后缓慢减速。",
                    "sets": 3,
                    "distance": "30-50米/组",
                    "rest": "90-120秒",
                    "tempo": "发力启动—匀速奔跑—缓慢减速",
                    "frequency": "每周1次",
                    "target": "强化推进力"
                },
                {
                    "name": "行进间高抬腿",
                    "description": "行进中交替高抬膝盖，强化髋屈肌和协调性",
                    "standard": "站立位，挺胸收腹，核心收紧。向前迈步的同时将另一侧膝盖向上高抬至大腿与地面平行（或更高），脚尖勾起。支撑腿前脚掌着地，保持身体弹性。双臂配合做跑步摆臂动作（前不露肘后不露手）。抬腿至最高点后主动下压落地，换另一侧。保持上半身直立不前倾，不后仰。落地轻盈，节奏稳定。",
                    "sets": 4,
                    "distance": "20-30米/组",
                    "rest": "45-60秒（走回起点）",
                    "tempo": "抬腿2秒，下压落地1秒，交替富有弹性",
                    "frequency": "每周2-3次",
                    "target": "提升髋屈肌力量和折叠效率"
                }
            ]
        }
    },
    "landing_quality": {
        "severe": {
            "condition": "score < 40",
            "advice": "着地方式存在严重问题，有较高的受伤风险。建议暂停跑步，从步行开始重新学习正确的着地模式。",
            "exercises": [
                {
                    "name": "靠墙静蹲",
                    "description": "背部靠墙，膝盖不超过脚尖，强化股四头肌",
                    "standard": "背靠墙壁站立，双脚前移约两步距离与肩同宽。沿墙壁缓慢下滑至大腿与地面呈90-120°（初学者从120°开始），膝盖不超过脚尖。保持背部贴墙，核心收紧。膝盖方向与脚尖方向一致，不外翻或内扣。均匀呼吸。",
                    "sets": 3,
                    "duration": "45-60秒/组",
                    "rest": "45-60秒",
                    "tempo": "匀速呼吸，全程保持姿势稳定",
                    "frequency": "每天",
                    "target": "强化股四头肌"
                },
                {
                    "name": "单腿硬拉",
                    "description": "单腿站立，身体前倾，改善平衡和髋部控制",
                    "standard": "单腿站立（可先扶墙辅助），另一腿微抬离地。保持背部挺直，核心收紧。以髋为轴向前折叠上半身，同时后腿向后伸直抬起，使身体呈T字形。感受支撑腿臀部和腘绳肌发力。到极限位置后缓慢回到起始位。全程保持髋部不旋转。",
                    "sets": 3,
                    "reps": "每侧8-10次",
                    "rest": "45-60秒",
                    "tempo": "前倾3秒，顶端停顿1秒，收回2秒",
                    "frequency": "每周3次",
                    "target": "改善平衡和髋部控制"
                },
                {
                    "name": "踮脚走",
                    "description": "前脚掌着地行走，学习正确着地模式",
                    "standard": "站立位，双脚与肩同宽。抬起脚跟，用前脚掌支撑体重。保持核心收紧，身体微前倾，用前脚掌向前迈进。每步落地时脚跟不触地，感受足弓的弹性。步幅适中，不要过大。双手可叉腰保持平衡。目光平视前方，不低头看脚。",
                    "sets": 3,
                    "distance": "15-20米/组",
                    "rest": "30-45秒",
                    "tempo": "匀速前进，每步轻盈有弹性",
                    "frequency": "每天",
                    "target": "改善着地模式"
                }
            ]
        },
        "moderate": {
            "condition": "score >= 40 and score < 60",
            "advice": "着地品质需要改进。标准要求：步幅适中，膝盖向前上方自然抬起，不过度伸直或锁死。落地时脚掌落在身体正下方，避免过度前伸，落地轻缓，以中足/前脚掌过渡为主，缓冲充分。支撑腿和摆动腿动作连贯，无甩小腿或拖地动作。",
            "exercises": [
                {
                    "name": "短步跑",
                    "description": "刻意缩短步幅跑步，让脚落在身体重心正下方",
                    "standard": "在平地或跑道上进行。用比正常步幅短30-40%的步幅慢跑，集中注意力感受每步落地位置在身体正下方（而非前方）。落地时膝盖微屈缓冲，前脚掌或中足先触地。保持躯干微前倾、核心收紧。双臂摆幅减小配合短步幅。",
                    "sets": 4,
                    "distance": "30-50米/组",
                    "rest": "60秒（慢走恢复）",
                    "tempo": "小步高频，落地轻盈，感受缓冲",
                    "frequency": "每周2次",
                    "target": "改善着地位置"
                },
                {
                    "name": "弹力带侧向行走",
                    "description": "膝盖套弹力带侧向行走，强化臀中肌",
                    "standard": "弹力带套在双膝上方或脚踝。站立位双脚与肩同宽，保持弹力带张力。微屈膝屈髋（约半蹲位），核心收紧，背部挺直。向侧方迈步，保持弹力带持续张力不松弛。每步先迈出一侧腿，另一侧再跟步至与肩同宽。左右方向各做一组。全程膝盖不内扣。",
                    "sets": 3,
                    "reps": "每方向15-20步",
                    "rest": "45-60秒",
                    "tempo": "迈步2秒，跟步2秒，匀速控制",
                    "frequency": "每周2次",
                    "target": "强化臀部外展肌"
                },
                {
                    "name": "泡沫轴放松",
                    "description": "使用泡沫轴放松小腿和大腿肌肉",
                    "standard": "坐姿或卧姿，将泡沫轴置于需要放松的肌肉下方（小腿后侧、大腿前侧/外侧/后侧）。利用自身体重施加压力，缓慢在肌肉上来回滚动。遇到酸痛点时停止按压约30秒至酸痛减轻。全程保持均匀深呼吸，肌肉放松不紧绷。",
                    "sets": 3,
                    "duration": "每个部位2-3分钟",
                    "rest": "无间隔，直接换下一个部位",
                    "tempo": "缓慢滚动，找到痛点后暂停按压",
                    "frequency": "每天",
                    "target": "改善肌肉紧张"
                }
            ]
        },
        "mild": {
            "condition": "score >= 60 and score < 80",
            "advice": "着地品质良好。可以通过一些微调来进一步优化，如尝试中前脚掌着地和减少垂直振幅。",
            "exercises": [
                {
                    "name": "跳绳",
                    "description": "快速跳绳，提升足部弹性和协调性",
                    "standard": "双手握绳末端，大臂夹紧身体两侧，手腕发力甩绳。跳跃时前脚掌着地，膝盖微屈缓冲，脚跟不触地。保持核心收紧，上半身稳定不晃动。落地轻盈，保持节奏。先从双脚跳开始，熟练后可尝试单脚交替跳。",
                    "sets": 4,
                    "duration": "1-2分钟/组",
                    "rest": "30-45秒",
                    "tempo": "匀速跳跃，保持稳定节奏",
                    "frequency": "每周2次",
                    "target": "提升足部弹性"
                },
                {
                    "name": "赤脚跑",
                    "description": "在安全场地赤脚慢跑，增强足部感知",
                    "standard": "选择平整柔软的草地、沙滩或塑胶跑道。脱鞋后先慢走2分钟让足部适应地面感觉。以极慢速度（比走路略快）开始小步跑，步幅缩短，用前脚掌或中足着地。集中注意力感受足底与地面的接触感，利用足部自然缓冲。全程保持警觉避开尖锐物。",
                    "sets": 2,
                    "duration": "5-8分钟/组",
                    "rest": "2-3分钟（穿鞋行走放松）",
                    "tempo": "慢速放松跑，重在感知而非速度",
                    "frequency": "每周1次",
                    "target": "增强足部感知"
                }
            ]
        }
    },
    "propulsion": {
        "severe": {
            "condition": "score < 40",
            "advice": "推进力严重不足，跑步效率很低。需要重点强化臀部和腿部力量，特别是臀大肌和腓肠肌。",
            "exercises": [
                {
                    "name": "负重深蹲",
                    "description": "杠铃深蹲，强化下肢综合力量",
                    "standard": "杠铃置于上斜方肌位置，挺胸收肩胛，核心收紧。双脚与肩同宽，脚尖微朝外。保持背部挺直，先屈髋再屈膝缓慢下蹲至大腿与地面平行或更低。膝盖方向与脚尖一致、不内扣。下蹲时吸气，站起时呼气。站起至顶端时夹紧臀部。",
                    "sets": 4,
                    "reps": "8-10次",
                    "rest": "90-120秒",
                    "tempo": "下蹲3秒，底部不停顿，站起2秒",
                    "frequency": "每周2次",
                    "target": "强化下肢力量"
                },
                {
                    "name": "罗马尼亚硬拉",
                    "description": "屈髋不屈膝的硬拉，强化腘绳肌和臀部",
                    "standard": "双手正握杠铃与肩同宽，站立时杠铃在大腿前侧。膝盖微屈，保持小腿垂直地面。以髋为轴向后推，上半身前倾，杠铃沿大腿前侧下放至膝盖下方（感受大腿后侧拉伸）。背部全程挺直、不弓背。收缩臀部发力将身体拉回直立位。",
                    "sets": 3,
                    "reps": "10-12次",
                    "rest": "60-90秒",
                    "tempo": "下放3秒，底部稍停，拉起2秒，顶端夹臀1秒",
                    "frequency": "每周2次",
                    "target": "强化腘绳肌和臀部"
                },
                {
                    "name": "提踵训练",
                    "description": "站立提踵，强化小腿腓肠肌",
                    "standard": "站立位，双脚与肩同宽，脚尖朝前。可双手扶墙或扶器械保持平衡。缓慢抬起脚跟至最高点，在顶端用力收缩小腿1-2秒，然后缓慢放回至脚跟低于水平面（充分拉伸）。全程膝盖伸直不弯曲。可进阶为单腿提踵增加强度。",
                    "sets": 4,
                    "reps": "15-20次",
                    "rest": "30-45秒",
                    "tempo": "上提2秒，顶端收缩2秒，下放3秒",
                    "frequency": "每周2次",
                    "target": "强化小腿肌肉"
                }
            ]
        },
        "moderate": {
            "condition": "score >= 40 and score < 60",
            "advice": "推进力需要加强。可以通过专门的力量训练和技术练习来提升后蹬力量。",
            "exercises": [
                {
                    "name": "臀桥",
                    "description": "仰卧臀部发力顶起身体，强化臀大肌",
                    "standard": "仰卧，双膝屈曲90°，双脚与肩同宽踩实地面。收紧核心，用臀部发力将髋部向上顶起至身体从肩膀到膝盖呈一条直线。在顶端用力夹紧臀部2秒，保持腹部收紧不弓腰。然后缓慢下放至接近地面但不完全触地，保持肌肉张力。",
                    "sets": 4,
                    "reps": "15-20次",
                    "rest": "45-60秒",
                    "tempo": "上顶2秒，顶端夹紧2秒，下放3秒",
                    "frequency": "每周3次",
                    "target": "强化臀大肌"
                },
                {
                    "name": "台阶上跳",
                    "description": "从地面跳上台阶，提升下肢爆发力",
                    "standard": "面对台阶（高度30-50cm，根据能力选择），站立位与台阶保持一步距离。屈膝屈髋下蹲至约半蹲位置，双臂后摆。利用下肢爆发力垂直跳上台阶，双脚同时落在台阶上，落地时屈膝缓冲。站稳后走回地面，不要跳下。全程核心收紧保持平衡。",
                    "sets": 3,
                    "reps": "8-10次",
                    "rest": "60-90秒",
                    "tempo": "下蹲→立刻跳起（无停顿），落地缓冲",
                    "frequency": "每周2次",
                    "target": "提升爆发力"
                },
                {
                    "name": "后蹬跑",
                    "description": "侧重后蹬发力的跑步技术练习",
                    "standard": "在平坦跑道上进行。与正常跑步不同，后蹬跑刻意强调每一步的向后蹬地发力。蹬地时充分伸展髋关节、膝关节和踝关节，感受臀部和腘绳肌发力。蹬地后小腿顺势折叠，然后膝盖前摆进入下一步。步幅比正常跑步稍大，步频稍低，重点感受后蹬力量。",
                    "sets": 4,
                    "distance": "30-50米/组",
                    "rest": "60-90秒（慢走恢复）",
                    "tempo": "有力蹬地→充分伸展→折叠前摆",
                    "frequency": "每周2次",
                    "target": "强化推进技术"
                }
            ]
        },
        "mild": {
            "condition": "score >= 60 and score < 80",
            "advice": "推进力良好。可以通过一些爆发力训练来进一步提升，如箱跳和冲刺练习。",
            "exercises": [
                {
                    "name": "箱跳",
                    "description": "双脚跳上箱子，提升全身爆发力",
                    "standard": "面对稳固的跳箱（高度40-60cm），站立位与箱保持一步距离。屈膝屈髋下蹲至约四分之一蹲，双臂后摆。用下肢爆发力向上跳起，双脚同时落在箱面上。落地时屈膝缓冲、核心收紧保持平衡。站稳后可以退下或跳下。跳下时落地同样要屈膝缓冲。",
                    "sets": 3,
                    "reps": "6-8次",
                    "rest": "90-120秒",
                    "tempo": "下蹲蓄力→爆发跳起→缓冲落地",
                    "frequency": "每周1次",
                    "target": "提升爆发力"
                },
                {
                    "name": "冲刺跑",
                    "description": "全力冲刺30-60米，提升最大速度和爆发力",
                    "standard": "选择50-80米平坦跑道。从站立起跑姿势或三点支撑起跑。启动后前几步用力蹬地加速，身体前倾角度较大。达到全速后保持高抬膝盖、有力摆臂、充分后蹬。冲刺过程中保持躯干稳定不左右摇晃。到终点后逐渐减速不停顿急停。注意充分热身后再进行。",
                    "sets": 4,
                    "distance": "30-60米/组",
                    "rest": "2-3分钟（充分恢复后再跑下一组）",
                    "tempo": "加速—全速保持—逐渐减速",
                    "frequency": "每周1次",
                    "target": "提升最大速度"
                }
            ]
        }
    },
    "symmetry": {
        "severe": {
            "condition": "score < 40",
            "advice": "左右对称性严重失衡，这会导致受伤风险大幅增加。建议进行单侧力量训练来纠正不平衡。",
            "exercises": [
                {
                    "name": "单腿深蹲",
                    "description": "单腿站立下蹲，平衡左右腿力量",
                    "standard": "单腿站立在平地或台阶边缘，另一腿向前伸直抬起。支撑腿缓慢下蹲至大腿与地面平行（或更低），膝盖方向与脚尖一致不内扣。下蹲时保持上半身挺直、核心收紧，不向一侧倾斜。蹲起时用臀部发力站起。先做弱侧再做强侧。",
                    "sets": 3,
                    "reps": "每侧6-8次（先从弱侧开始）",
                    "rest": "45-60秒（两侧之间不休息）",
                    "tempo": "下蹲3秒，底部不停顿，站起2秒",
                    "frequency": "每周2次",
                    "target": "平衡左右腿力量"
                },
                {
                    "name": "单腿臀桥",
                    "description": "单腿进行臀桥，平衡臀部力量",
                    "standard": "仰卧，双膝屈曲90°。将一只脚抬离地面伸直，另一只脚踩实地面。用踩地脚跟发力将髋部向上顶起至身体呈直线，在顶端夹紧臀部2秒。缓慢下放至接近地面后再次发力。全程核心收紧不弓腰，保持骨盆水平不旋转。",
                    "sets": 3,
                    "reps": "每侧10-12次（先做弱侧）",
                    "rest": "45-60秒（两侧之间不休息）",
                    "tempo": "上顶2秒，顶端夹紧2秒，下放3秒",
                    "frequency": "每周3次",
                    "target": "平衡臀部力量"
                },
                {
                    "name": "单腿提踵",
                    "description": "单腿站立提踵，平衡小腿力量",
                    "standard": "单腿站立，双手扶墙或扶器械保持平衡。缓慢抬起脚跟至最高点，在顶端用力收缩小腿1-2秒，然后缓慢下放至脚跟低于水平面（充分拉伸）。全程膝盖伸直不弯曲。先做弱侧再做强侧。保持身体直立不倾斜。",
                    "sets": 4,
                    "reps": "每侧12-15次（先做弱侧）",
                    "rest": "30-45秒（两侧之间不休息）",
                    "tempo": "上提2秒，顶端收缩2秒，下放3秒",
                    "frequency": "每周2次",
                    "target": "平衡小腿力量"
                }
            ]
        },
        "moderate": {
            "condition": "score >= 40 and score < 60",
            "advice": "左右对称性存在偏差。摆臂应以肩关节为轴前后自然摆动，不左右横摆、不交叉过中线、不甩手。肘部保持约90°夹角，摆幅控制在腰侧至胸前之间，前不露肘、后不露手。建议进行单侧力量训练改善不平衡。",
            "exercises": [
                {
                    "name": "单侧罗马尼亚硬拉",
                    "description": "单腿进行硬拉，平衡髋部力量",
                    "standard": "单腿站立，另一腿微抬离地。保持支撑腿微屈膝，背部挺直。手持哑铃（初学者2-8kg），以髋为轴向前折叠上半身，哑铃沿腿部前侧下放至膝盖下方。感受支撑腿腘绳肌拉伸。用臀部发力拉回直立位。全程髋部不旋转。先做弱侧再做强侧。",
                    "sets": 3,
                    "reps": "每侧8-10次（先做弱侧）",
                    "rest": "45-60秒（两侧之间不休息）",
                    "tempo": "下放3秒，底部稍停，拉起2秒",
                    "frequency": "每周2次",
                    "target": "平衡髋部力量"
                },
                {
                    "name": "侧卧抬腿",
                    "description": "侧卧抬起上方腿，强化臀部外展肌",
                    "standard": "侧卧于垫上，下方腿微屈以保持稳定。上方腿伸直与身体呈一直线，脚尖朝前。收紧核心保持骨盆稳定不前后滚动。用臀部外侧发力将上方腿抬起至约45°，在顶端停顿1-2秒后缓慢放下。全程身体不晃动、不上翻。做完一侧换另一侧。",
                    "sets": 3,
                    "reps": "每侧15-20次（先做弱侧）",
                    "rest": "30-45秒（两侧之间不休息）",
                    "tempo": "上抬2秒，顶端停顿1-2秒，下放3秒",
                    "frequency": "每周2次",
                    "target": "强化臀部外展"
                },
                {
                    "name": "坐姿摆臂",
                    "description": "坐姿专注于摆臂动作练习",
                    "standard": "坐于凳子上，上身挺直，核心收紧。双肘屈曲90°，大臂贴近身体两侧不张开。以肩关节为轴前后摆动双臂，前摆到手在胸前高度，后摆到手超过体侧。全程保持肘部90°不变，双肩放松不耸起。先慢速做标准动作，再逐渐加速到跑步节奏。",
                    "sets": 3,
                    "reps": "20-30次",
                    "rest": "30-45秒",
                    "tempo": "前摆后摆各1秒，匀速控制",
                    "frequency": "每周2次",
                    "target": "改善摆臂对称性"
                }
            ]
        },
        "mild": {
            "condition": "score >= 60 and score < 80",
            "advice": "对称性良好。可以通过一些功能性训练来维持和进一步优化双侧协调性。",
            "exercises": [
                {
                    "name": "波比跳",
                    "description": "全身协调性训练，提升心肺和协调能力",
                    "standard": "站立位开始，下蹲双手撑地（在脚前），双腿向后跳成平板支撑位（做一个俯卧撑可选），双腿向前跳回蹲位，垂直向上跳起同时双手举过头顶。落地时屈膝缓冲。全程核心收紧，动作连贯不断。可省略俯卧撑降低难度。",
                    "sets": 3,
                    "reps": "10-15次",
                    "rest": "60-90秒",
                    "tempo": "下蹲—平板—跳回—起跳，连贯完成",
                    "frequency": "每周1次",
                    "target": "提升全身协调"
                },
                {
                    "name": "平板支撑交替抬手",
                    "description": "平板支撑时交替抬手，改善核心平衡",
                    "standard": "平板支撑姿势：双肘在肩关节正下方撑地，双脚与肩同宽。核心收紧，身体呈一条直线。保持身体稳定不晃动，缓慢将右手抬离地面向前伸直（或轻触对侧肩膀），停顿1-2秒后放回。换左手重复。交替进行。全程髋部不旋转、不塌腰。",
                    "sets": 3,
                    "reps": "每侧8-10次",
                    "rest": "45-60秒",
                    "tempo": "抬手2秒，停顿1-2秒，收回2秒",
                    "frequency": "每周2次",
                    "target": "改善核心平衡"
                }
            ]
        }
    }
}

# ==================== 辅助函数 ====================
def get_standard_pose(running_type, gender, view='side_view'):
    """获取标准跑姿数据"""
    return STANDARD_POSE_LIBRARY.get(running_type, {}).get(gender, {}).get(view, {})

def detect_running_type(params):
    """根据参数自动检测跑步类型"""
    knee_min_angle = params.get('knee_min_angle', 90)
    stride_ratio = params.get('stride_ratio', 1.5)
    
    if knee_min_angle < 50 and stride_ratio > 2.0:
        return 'sprint'
    elif 50 <= knee_min_angle < 70 and 1.3 <= stride_ratio < 1.8:
        return 'middle_distance'
    elif 60 <= knee_min_angle < 80 and 1.1 <= stride_ratio < 1.4:
        return 'jogging'
    elif knee_min_angle > 150 and stride_ratio < 1.2:
        return 'walking'
    else:
        return 'middle_distance'  # 默认

def detect_gender(params):
    """根据骨骼比例检测性别"""
    shoulder_hip_ratio = params.get('shoulder_hip_ratio', 1.1)
    
    if shoulder_hip_ratio > 1.15:
        return 'male'
    else:
        return 'female'

def generate_training_plan(issues):
    """根据检测问题生成训练计划"""
    plan = {
        "phase1": {  # 技术感知期 第1-2周
            "name": "技术感知期",
            "duration": "第1-2周",
            "goal": "建立正确的动作感知",
            "focus": ["核心稳定", "着地模式"],
            "frequency": "每天15-20分钟",
            "exercises": []
        },
        "phase2": {  # 技术强化期 第3-4周
            "name": "技术强化期",
            "duration": "第3-4周",
            "goal": "在跑步中应用新技术",
            "focus": ["下肢力学", "推进力"],
            "frequency": "每周3次跑步+2次力量",
            "exercises": []
        },
        "phase3": {  # 技术巩固期 第5-6周
            "name": "技术巩固期",
            "duration": "第5-6周",
            "goal": "让新技术成为习惯",
            "focus": ["整体协调", "对称性"],
            "frequency": "正常训练强度，每2周复查",
            "exercises": []
        }
    }
    
    for issue in issues:
        dim_key = issue.get('dimension')
        level = issue.get('level')
        
        if dim_key in OPTIMIZATION_RULES and level in OPTIMIZATION_RULES[dim_key]:
            exercises = OPTIMIZATION_RULES[dim_key][level].get('exercises', [])
            
            if level == 'severe':
                plan["phase1"]["exercises"].extend(exercises)
            elif level == 'moderate':
                plan["phase2"]["exercises"].extend(exercises)
            else:
                plan["phase3"]["exercises"].extend(exercises)
    
    # 去重练习
    for phase in plan:
        if 'exercises' in plan[phase]:
            unique_exercises = []
            seen_names = set()
            for ex in plan[phase]['exercises']:
                if ex['name'] not in seen_names:
                    seen_names.add(ex['name'])
                    unique_exercises.append(ex)
            plan[phase]['exercises'] = unique_exercises
    
    return plan

def calculate_gaussian_score(value, ideal_center, tolerance):
    """高斯型评分函数"""
    deviation = abs(value - ideal_center)
    score = 100 * np.exp(-(deviation ** 2) / (2 * tolerance ** 2))
    return max(0, min(100, score))

def calculate_cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_pearson_correlation(vec1, vec2):
    """计算皮尔逊相关系数"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if len(vec1) < 2 or len(vec2) < 2:
        return 0.0
    
    min_len = min(len(vec1), len(vec2))
    vec1_norm = vec1[:min_len]
    vec2_norm = vec2[:min_len]
    
    mean1 = np.mean(vec1_norm)
    mean2 = np.mean(vec2_norm)
    
    std1 = np.std(vec1_norm)
    std2 = np.std(vec2_norm)
    
    if std1 == 0 or std2 == 0:
        return 0.0
    
    covariance = np.mean((vec1_norm - mean1) * (vec2_norm - mean2))
    correlation = covariance / (std1 * std2)
    
    return max(-1.0, min(1.0, correlation))

def get_optimization_rules(dim_key):
    """获取指定维度的优化规则"""
    return OPTIMIZATION_RULES.get(dim_key, {})