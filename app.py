# encoding: utf-8
import os
import cv2 as cv
import numpy as np
import base64
import time
import json
import tempfile
import hashlib
import secrets
from flask import Flask, render_template, Response, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
_BASEDIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(_BASEDIR, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(_BASEDIR, 'results')
# 每次关闭浏览器或重启服务器后都需要重新登录
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_COOKIE_MAX_AGE'] = None       # 浏览器关闭即失效
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
USERS_FILE = os.path.join(_BASEDIR, 'users.json')
CORS(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

# 使用短ASCII临时路径存放上传视频（避免中文路径导致OpenCV读取失败）
import tempfile
_UPLOAD_TEMP = os.path.join(tempfile.gettempdir(), 'pose_uploads')
if not os.path.exists(_UPLOAD_TEMP):
    os.makedirs(_UPLOAD_TEMP)

detector = None
processing_lock = False

def load_model():
    global detector
    print("正在加载人体姿态检测模型...")
    
    try:
        from pose_detector import PoseDetector
        # MediaPipe 内部 C++ API 不支持中文路径，将模型复制到临时目录
        model_src = os.path.join(os.path.dirname(__file__), 'pose_landmarker.task')
        if os.path.exists(model_src):
            import shutil
            import tempfile
            model_dir = os.path.join(tempfile.gettempdir(), 'pose_models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'pose_landmarker.task')
            if not os.path.exists(model_path):
                shutil.copy2(model_src, model_path)
                print(f"模型已复制到: {model_path}")
            else:
                print(f"模型已存在: {model_path}")
        else:
            print(f"模型文件不存在: {model_src}")
            model_path = 'pose_landmarker.task'
        detector = PoseDetector(model_path=model_path, maxPoses=1)
        
        if detector.is_loaded():
            print("模型加载成功！")
            return True
        else:
            print("模型加载失败：检测器未成功初始化")
            return False
    except ImportError as e:
        print(f"导入失败: {e}")
        return False
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

# ==================== 用户认证 ====================

def _load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

@app.route('/')
def splash():
    return render_template('splash.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/main')
def index():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    if len(username) < 2 or len(password) < 4:
        return jsonify({'status': 'error', 'message': '用户名至少2个字符，密码至少4个字符'})
    users = _load_users()
    if username in users:
        return jsonify({'status': 'error', 'message': '用户名已存在'})
    users[username] = {
        'password': hashlib.sha256(password.encode()).hexdigest(),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    _save_users(users)
    session['user'] = username
    return jsonify({'status': 'success', 'message': '注册成功'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    users = _load_users()
    user = users.get(username)
    if not user or user['password'] != hashlib.sha256(password.encode()).hexdigest():
        return jsonify({'status': 'error', 'message': '用户名或密码错误'})
    session['user'] = username
    return jsonify({'status': 'success', 'message': '登录成功'})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'status': 'success', 'message': '已退出'})

@app.route('/api/check_session', methods=['GET'])
def check_session():
    return jsonify({'authenticated': 'user' in session, 'user': session.get('user')})

@app.route('/api/status', methods=['GET'])
def get_status():
    global detector
    model_loaded = detector.is_loaded() if detector else False
    return jsonify({
        'status': 'ready',
        'model_loaded': model_loaded,
        'processing': processing_lock
    })

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global processing_lock
    if processing_lock:
        return jsonify({'status': 'error', 'message': '系统正在处理其他视频，请稍候'})

    try:
        video_type = request.form.get('video_type')
        if video_type not in ['standard', 'test', 'before', 'after']:
            return jsonify({'status': 'error', 'message': '无效的视频类型'})

        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': '没有上传视频'})

        file = request.files['video']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择视频文件'})

        processing_lock = True
        file_path = os.path.join(_UPLOAD_TEMP, f'{video_type}_video.mp4')
        file.save(file_path)

        type_names = {'standard': '标准', 'test': '测试', 'before': '以前的', 'after': '现在的'}
        return jsonify({
            'status': 'success',
            'message': f'{type_names.get(video_type, "")}视频上传成功',
            'video_path': file_path
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'上传失败: {str(e)}'})
    finally:
        processing_lock = False

@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    global detector, processing_lock
    
    if processing_lock:
        return jsonify({'status': 'error', 'message': '系统正在处理其他视频，请稍候'})
    
    if not detector or not detector.is_loaded():
        return jsonify({'status': 'error', 'message': '模型未加载，请检查模型文件'})

    processing_lock = True
    try:
        data = request.json
        standard_video = data.get('standard_video')
        test_video = data.get('test_video')

        if not standard_video or not test_video:
            return jsonify({'status': 'error', 'message': '请先上传标准视频和测试视频'})

        standard_cap = cv.VideoCapture(standard_video)
        test_cap = cv.VideoCapture(test_video)

        if not standard_cap.isOpened():
            return jsonify({'status': 'error', 'message': '无法打开标准视频'})
        if not test_cap.isOpened():
            return jsonify({'status': 'error', 'message': '无法打开测试视频'})

        standard_params_list = []
        test_params_list = []
        frame_count = 0
        max_frames = min(int(standard_cap.get(cv.CAP_PROP_FRAME_COUNT)),
                        int(test_cap.get(cv.CAP_PROP_FRAME_COUNT)), 100)

        for i in range(max_frames):
            ret_std, frame_std = standard_cap.read()
            ret_test, frame_test = test_cap.read()

            if not ret_std or not ret_test:
                break

            if i % 3 != 0:
                continue

            _, _, std_landmarks = detector.find_pose_landmarks(frame_std, draw=False)
            _, _, test_landmarks = detector.find_pose_landmarks(frame_test, draw=False)

            std_params = detector.get_key_parameters(std_landmarks)
            test_params = detector.get_key_parameters(test_landmarks)

            if std_params:
                standard_params_list.append(std_params)
            if test_params:
                test_params_list.append(test_params)

            frame_count += 1

        standard_cap.release()
        test_cap.release()

        if not standard_params_list:
            return jsonify({'status': 'error', 'message': '标准视频中未检测到人体姿态，请确保视频中有清晰的人体'})

        avg_standard_params = {}
        if standard_params_list:
            for key in standard_params_list[0].keys():
                if key != 'all_angles':
                    avg_standard_params[key] = sum(p.get(key, 0) for p in standard_params_list) / len(standard_params_list)

        result_frames = []
        economy_percentages = []
        phase_scores = {}

        standard_cap = cv.VideoCapture(standard_video)
        test_cap = cv.VideoCapture(test_video)

        standard_frames = []
        test_frames = []
        
        for i in range(min(max_frames, 60)):
            ret_std, frame_std = standard_cap.read()
            ret_test, frame_test = test_cap.read()

            if not ret_std or not ret_test:
                break

            standard_frames.append(frame_std.copy())
            test_frames.append(frame_test.copy())

        standard_cap.release()
        test_cap.release()

        std_phases, std_phase_frames = detector.extract_gait_phases(standard_frames)
        test_phases, test_phase_frames = detector.extract_gait_phases(test_frames)
        
        aligned_pairs = detector.align_gait_phases(std_phases, test_phases)
        
        phase_matched_frames = {}
        for std_idx, test_idx, phase_key in aligned_pairs:
            if phase_key not in phase_matched_frames:
                phase_matched_frames[phase_key] = []
            phase_matched_frames[phase_key].append((std_idx, test_idx))
        
        for phase_key, frame_pairs in phase_matched_frames.items():
            for std_idx, test_idx in frame_pairs:
                if std_idx < len(standard_frames) and test_idx < len(test_frames):
                    frame_std = standard_frames[std_idx]
                    frame_test = test_frames[test_idx]
                    
                    processed_std, skeleton_std, std_landmarks = detector.find_pose_landmarks(frame_std, draw=True)
                    processed_test, skeleton_test, test_landmarks = detector.find_pose_landmarks(frame_test, draw=True)

                    combined = detector.frame_combine(processed_std, processed_test)

                    _, combined_encoded = cv.imencode('.jpg', combined, [int(cv.IMWRITE_JPEG_QUALITY), 80])
                    frame_base64 = base64.b64encode(combined_encoded).decode('utf-8')
                    result_frames.append({
                        'frame': frame_base64,
                        'phase': phase_key,
                        'std_frame': std_idx,
                        'test_frame': test_idx
                    })

                    if std_landmarks and test_landmarks:
                        std_params = detector.get_key_parameters(std_landmarks)
                        test_params = detector.get_key_parameters(test_landmarks)
                        if std_params and test_params:
                            economy = detector.compare_poses(std_params, test_params)
                            economy_percentages.append(economy)
                            
                            if phase_key not in phase_scores:
                                phase_scores[phase_key] = []
                            phase_scores[phase_key].append(economy)
        
        if not economy_percentages:
            for i in range(min(len(standard_frames), len(test_frames), 10)):
                frame_std = standard_frames[i]
                frame_test = test_frames[i]
                
                processed_std, skeleton_std, std_landmarks = detector.find_pose_landmarks(frame_std, draw=True)
                processed_test, skeleton_test, test_landmarks = detector.find_pose_landmarks(frame_test, draw=True)

                combined = detector.frame_combine(processed_std, processed_test)

                _, combined_encoded = cv.imencode('.jpg', combined, [int(cv.IMWRITE_JPEG_QUALITY), 80])
                frame_base64 = base64.b64encode(combined_encoded).decode('utf-8')
                result_frames.append({
                    'frame': frame_base64,
                    'phase': 'unknown',
                    'std_frame': i,
                    'test_frame': i
                })

                if std_landmarks and test_landmarks:
                    std_params = detector.get_key_parameters(std_landmarks)
                    test_params = detector.get_key_parameters(test_landmarks)
                    if std_params and test_params:
                        economy = detector.compare_poses(std_params, test_params)
                        economy_percentages.append(economy)

        final_economy = sum(economy_percentages) / len(economy_percentages) if economy_percentages else 0
        
        phase_summary = {}
        for phase_key, scores in phase_scores.items():
            if scores:
                phase_summary[phase_key] = {
                    'avg_score': round(sum(scores) / len(scores), 1),
                    'count': len(scores)
                }
        
        std_curves = detector.extract_motion_curve(standard_frames)
        test_curves = detector.extract_motion_curve(test_frames)
        
        curve_result = detector.compare_poses(std_curves, test_curves)
        
        final_score = final_economy
        fatal_errors = []
        
        if isinstance(curve_result, dict):
            final_score = curve_result.get('total_score', final_economy)
            fatal_errors = curve_result.get('fatal_errors', [])
        elif isinstance(curve_result, float):
            final_score = curve_result

        return jsonify({
            'status': 'success',
            'frames': [f['frame'] for f in result_frames],
            'economy_percentage': round(final_score, 1),
            'standard_params': {k: round(v, 1) for k, v in avg_standard_params.items()},
            'frames_count': len(result_frames),
            'phase_alignment': {
                'total_matched': len(aligned_pairs),
                'phases': phase_summary
            },
            'fatal_errors': fatal_errors,
            'score_type': 'standard_vs_test'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'})
    finally:
        processing_lock = False

@app.route('/api/process_video_stream', methods=['POST'])
def process_video_stream():
    global detector, processing_lock
    
    if processing_lock:
        return jsonify({'status': 'error', 'message': '系统正在处理其他视频，请稍候'})
    
    if not detector or not detector.is_loaded():
        return jsonify({'status': 'error', 'message': '模型未加载'})

    processing_lock = True
    try:
        video_type = request.form.get('video_type')
        if video_type not in ['standard', 'test']:
            return jsonify({'status': 'error', 'message': '无效的视频类型'})

        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': '没有上传视频'})

        file = request.files['video']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择视频文件'})

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_path = tmp_file.name
            file.save(tmp_path)

        cap = cv.VideoCapture(tmp_path)
        if not cap.isOpened():
            os.unlink(tmp_path)
            return jsonify({'status': 'error', 'message': '无法打开视频文件'})

        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = 0
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        params_list = []

        preview_frame = None
        preview_encoded = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed, skeleton, landmarks = detector.find_pose_landmarks(frame, draw=True)

            if landmarks:
                params = detector.get_key_parameters(landmarks)
                if params:
                    params_list.append(params)

            # 获取预览帧（第一帧即可）
            if preview_frame is None:
                _, encoded = cv.imencode('.jpg', processed, [int(cv.IMWRITE_JPEG_QUALITY), 70])
                preview_encoded = base64.b64encode(encoded).decode('utf-8')
                preview_frame = True  # 标记已获取

            frame_count += 1

        cap.release()
        os.unlink(tmp_path)

        if preview_encoded:
            return jsonify({
                'status': 'success',
                'video_type': video_type,
                'frame': preview_encoded,
                'frame_count': frame_count,
                'total_frames': total_frames,
                'detected': bool(landmarks),
                'detected_count': len(params_list)
            })

        return jsonify({
            'status': 'success',
            'video_type': video_type,
            'message': '视频处理完成',
            'frame_count': frame_count,
            'detected_count': len(params_list)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'处理失败: {str(e)}'})
    finally:
        processing_lock = False

@app.route('/api/compare_videos', methods=['POST'])
def compare_videos():
    global detector, processing_lock
    
    if processing_lock:
        return jsonify({'status': 'error', 'message': '系统正在处理，请稍候'})
    
    if not detector or not detector.is_loaded():
        return jsonify({'status': 'error', 'message': '模型未加载，请检查模型文件'})

    processing_lock = True
    try:
        data = request.json
        standard_video = data.get('standard_video')
        test_video = data.get('test_video')

        if not standard_video or not test_video:
            return jsonify({'status': 'error', 'message': '请上传两个视频'})

        cap_std = cv.VideoCapture(standard_video)
        cap_test = cv.VideoCapture(test_video)

        if not cap_std.isOpened() or not cap_test.isOpened():
            return jsonify({'status': 'error', 'message': '无法打开视频'})

        std_params_accum = None
        test_params_accum = None
        frame_count = 0
        max_frames = min(int(cap_std.get(cv.CAP_PROP_FRAME_COUNT)),
                        int(cap_test.get(cv.CAP_PROP_FRAME_COUNT)), 150)

        while frame_count < max_frames:
            ret_std, frame_std = cap_std.read()
            ret_test, frame_test = cap_test.read()

            if not ret_std or not ret_test:
                break

            if frame_count % 4 == 0:
                _, _, std_landmarks = detector.find_pose_landmarks(frame_std, draw=False)
                _, _, test_landmarks = detector.find_pose_landmarks(frame_test, draw=False)

                std_params = detector.get_key_parameters(std_landmarks)
                test_params = detector.get_key_parameters(test_landmarks)

                if std_params:
                    if std_params_accum is None:
                        std_params_accum = {k: [] for k in std_params.keys()}
                    for k, v in std_params.items():
                        if k != 'all_angles':
                            std_params_accum[k].append(v)

                if test_params:
                    if test_params_accum is None:
                        test_params_accum = {k: [] for k in test_params.keys()}
                    for k, v in test_params.items():
                        if k != 'all_angles':
                            test_params_accum[k].append(v)

            frame_count += 1

        cap_std.release()
        cap_test.release()

        if not std_params_accum:
            return jsonify({'status': 'error', 'message': '标准视频未检测到人体姿态，请确保视频中有清晰的人体'})

        avg_std = {}
        for k, v in std_params_accum.items():
            if len(v) > 0:
                avg_std[k] = sum(v) / len(v)
        
        avg_test = {}
        if test_params_accum:
            for k, v in test_params_accum.items():
                if len(v) > 0:
                    avg_test[k] = sum(v) / len(v)
        else:
            avg_test = avg_std.copy()

        economy = detector.compare_poses(avg_std, avg_test)

        param_diffs = {}
        if avg_std and avg_test:
            for key in avg_std.keys():
                if key not in ['all_angles']:
                    diff = abs(avg_std[key] - avg_test.get(key, avg_std[key]))
                    param_diffs[key] = round(diff, 1)

        comparison_frames = []
        cap_std = cv.VideoCapture(standard_video)
        cap_test = cv.VideoCapture(test_video)

        for i in range(min(frame_count, 20)):
            ret_std, frame_std = cap_std.read()
            ret_test, frame_test = cap_test.read()

            if not ret_std or not ret_test:
                break

            processed_std, _, std_landmarks = detector.find_pose_landmarks(frame_std, draw=True)
            processed_test, _, test_landmarks = detector.find_pose_landmarks(frame_test, draw=True)

            combined = detector.frame_combine(processed_std, processed_test)

            _, combined_encoded = cv.imencode('.jpg', combined, [int(cv.IMWRITE_JPEG_QUALITY), 75])
            comparison_frames.append(base64.b64encode(combined_encoded).decode('utf-8'))

        cap_std.release()
        cap_test.release()

        return jsonify({
            'status': 'success',
            'economy_percentage': economy,
            'standard_params': {k: round(v, 1) for k, v in avg_std.items()},
            'test_params': {k: round(v, 1) for k, v in avg_test.items()},
            'param_differences': param_diffs,
            'comparison_frames': comparison_frames,
            'analyzed_frames': frame_count
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'对比分析失败: {str(e)}'})
    finally:
        processing_lock = False

@app.route('/api/clear_videos', methods=['POST'])
def clear_videos():
    global processing_lock
    processing_lock = False
    try:
        for filename in os.listdir(_UPLOAD_TEMP):
            file_path = os.path.join(_UPLOAD_TEMP, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({'status': 'success', 'message': '已清除所有上传视频'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'清除失败: {str(e)}'})

# ==================== 雷达图分析接口 ====================

@app.route('/api/analyze_radar', methods=['POST'])
def analyze_radar():
    global detector, processing_lock
    
    if processing_lock:
        return jsonify({'status': 'error', 'message': '系统正在处理，请稍候'})
    
    if not detector or not detector.is_loaded():
        return jsonify({'status': 'error', 'message': '模型未加载'})
    
    processing_lock = True
    try:
        data = request.json
        video_path = data.get('video_path')
        
        if not video_path:
            return jsonify({'status': 'error', 'message': '请提供视频路径'})
        
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'status': 'error', 'message': '无法打开视频'})

        fps = cap.get(cv.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        max_read = min(total_frames, 2700)
        max_keep = 900
        frame_list = []

        for i in range(max_read):
            ret, frame = cap.read()
            if not ret:
                break
            if i % 3 == 0 and len(frame_list) < max_keep:
                frame_list.append(frame.copy())

        cap.release()

        if not frame_list:
            return jsonify({'status': 'error', 'message': '无法读取视频帧'})

        radar_result, skeleton_data = detector.analyze_video_with_skeleton(
            frame_list, max_radar_frames=60
        )

        if not radar_result:
            return jsonify({'status': 'error', 'message': '分析失败'})

        return jsonify({
            'status': 'success',
            'radar_scores': radar_result['radar_scores'],
            'total_score': radar_result['total_score'],
            'running_type': radar_result['running_type'],
            'gender': radar_result['gender'],
            'params_summary': radar_result['params_summary'],
            'issues': radar_result['issues'],
            'optimization_plan': radar_result['optimization_plan'],
            'phase_scores': radar_result['phase_scores'],
            'skeleton_data': skeleton_data,
            'frame_count': len(frame_list)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'})
    finally:
        processing_lock = False

@app.route('/api/get_radar_dimensions', methods=['GET'])
def get_radar_dimensions():
    """获取雷达图维度定义"""
    from standard_pose_library import RADAR_DIMENSIONS
    return jsonify({
        'status': 'success',
        'dimensions': RADAR_DIMENSIONS
    })

# 模拟历史记录存储（实际应用中应使用数据库）
history_records = []

@app.route('/api/save_result', methods=['POST'])
def save_result():
    """保存分析结果到历史记录"""
    try:
        data = request.json
        record = {
            'id': len(history_records) + 1,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_score': data.get('total_score'),
            'radar_scores': data.get('radar_scores'),
            'running_type': data.get('running_type'),
            'gender': data.get('gender'),
            'params_summary': data.get('params_summary'),
            'phase_scores': data.get('phase_scores'),
            'issues': data.get('issues'),
            'optimization_plan': data.get('optimization_plan')
        }
        history_records.append(record)
        return jsonify({'status': 'success', 'record_id': record['id']})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'保存失败: {str(e)}'})

@app.route('/api/get_history', methods=['GET'])
def get_history():
    """获取历史记录"""
    return jsonify({
        'status': 'success',
        'records': history_records[-5:]  # 返回最近5条记录
    })

@app.route('/api/get_history_by_id/<int:record_id>', methods=['GET'])
def get_history_by_id(record_id):
    """根据ID获取单条历史记录"""
    record = next((r for r in history_records if r['id'] == record_id), None)
    if record:
        return jsonify({'status': 'success', 'record': record})
    return jsonify({'status': 'error', 'message': '记录不存在'})

@app.route('/api/get_comparison_history', methods=['GET'])
def get_comparison_history():
    """获取用于对比的历史记录（最近3条）"""
    if len(history_records) < 2:
        return jsonify({'status': 'success', 'records': [], 'message': '历史记录不足，无法对比'})
    
    recent_records = history_records[-3:]
    return jsonify({
        'status': 'success',
        'records': recent_records,
        'current_record': recent_records[-1] if recent_records else None
    })

@app.route('/api/get_standard_pose', methods=['GET'])
def get_standard_pose():
    """获取标准跑姿库数据"""
    from standard_pose_library import STANDARD_POSE_LIBRARY, RADAR_DIMENSIONS
    running_type = request.args.get('type', 'middle_distance')
    gender = request.args.get('gender', 'male')
    
    return jsonify({
        'status': 'success',
        'standard_data': STANDARD_POSE_LIBRARY.get(running_type, {}).get(gender, {}),
        'dimensions': RADAR_DIMENSIONS
    })

@app.route('/api/get_running_types', methods=['GET'])
def get_running_types():
    """获取所有跑步类型列表"""
    from standard_pose_library import STANDARD_POSE_LIBRARY
    types = list(STANDARD_POSE_LIBRARY.keys())
    return jsonify({
        'status': 'success',
        'types': types,
        'type_labels': {
            'sprint': '短跑',
            'middle_distance': '中长跑',
            'jogging': '慢跑',
            'walking': '竞走'
        }
    })

def _run_single_video_analysis(detector, video_path, running_type=None, gender=None):
    """分析单个视频，返回结构化结果（不含status/message）"""
    # 处理中文路径：复制到临时目录确保 OpenCV 可读
    import tempfile
    import shutil
    temp_video = None
    try:
        if any(ord(c) > 127 for c in video_path):
            ext = os.path.splitext(video_path)[1] or '.mp4'
            tmpdir = tempfile.mkdtemp(prefix='pose_')
            temp_video = os.path.join(tmpdir, f'video{ext}')
            shutil.copy2(video_path, temp_video)
            cap = cv.VideoCapture(temp_video)
        else:
            cap = cv.VideoCapture(video_path)
    except Exception:
        cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        if temp_video:
            shutil.rmtree(os.path.dirname(temp_video), ignore_errors=True)
        return None

    frame_list = []
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    max_keep = 900

    # 兼容 CAP_PROP_FRAME_COUNT 返回 0 或负数的情况（常见于 Windows）
    if total_frames > 0:
        max_read = min(total_frames, 2700)
        for i in range(max_read):
            ret, frame = cap.read()
            if not ret:
                break
            if i % 3 == 0 and len(frame_list) < max_keep:
                frame_list.append(frame.copy())
    else:
        # 降级方案：直接逐帧读到够为止
        read_count = 0
        while len(frame_list) < max_keep:
            ret, frame = cap.read()
            if not ret:
                break
            if read_count % 3 == 0:
                frame_list.append(frame.copy())
            read_count += 1
            if read_count > 9000:  # 安全上限
                break

    cap.release()
    if temp_video:
        shutil.rmtree(os.path.dirname(temp_video), ignore_errors=True)

    if not frame_list:
        return None

    radar_result, skeleton_data = detector.analyze_video_with_skeleton(
        frame_list, running_type, gender, max_radar_frames=60
    )

    if not radar_result:
        return None

    # 生成关键帧预览图
    frame_previews = []
    total_frames = len(frame_list)
    if total_frames > 0:
        preview_count = min(20, total_frames)
        sample_indices = [int(i * total_frames / preview_count) for i in range(preview_count)]
        for idx in sample_indices:
            frame = frame_list[idx]
            processed, _, _ = detector.find_pose_landmarks(frame, draw=True)
            _, buffer = cv.imencode('.jpg', processed, [int(cv.IMWRITE_JPEG_QUALITY), 80])
            frame_previews.append(base64.b64encode(buffer).decode('utf-8'))

    return {
        'radar_scores': radar_result['radar_scores'],
        'total_score': radar_result['total_score'],
        'running_type': radar_result['running_type'],
        'gender': radar_result['gender'],
        'params_summary': radar_result['params_summary'],
        'issues': radar_result['issues'],
        'specific_errors': radar_result.get('specific_errors', []),
        'optimization_plan': radar_result['optimization_plan'],
        'phase_scores': radar_result['phase_scores'],
        'skeleton_data': skeleton_data,
        'frame_previews': frame_previews,
        'frame_count': len(frame_list),
        'fps': fps
    }


def _generate_progress_report(before, after):
    """对比训练前后的分析结果，生成进步报告"""
    # --- 雷达维度对比 ---
    dim_config = [
        ('core_stability', '核心稳定度'),
        ('leg_fold_efficiency', '腿部折叠效率'),
        ('landing_quality', '落地质量'),
        ('propulsion', '推进力'),
        ('symmetry', '左右对称性'),
    ]
    radar_comparison = []
    for key, name in dim_config:
        bs = before['radar_scores'].get(key, 0)
        aft = after['radar_scores'].get(key, 0)
        change = round(aft - bs, 1)
        if change > 2:
            direction = 'improved'
        elif change < -2:
            direction = 'declined'
        else:
            direction = 'unchanged'
        radar_comparison.append({
            'dimension_key': key,
            'dimension_name': name,
            'before_score': round(bs, 1),
            'after_score': round(aft, 1),
            'change': change,
            'direction': direction
        })

    # --- 总体分数变化 ---
    before_total = before['total_score']
    after_total = after['total_score']
    total_change = round(after_total - before_total, 1)
    if total_change > 2:
        total_direction = 'improved'
    elif total_change < -2:
        total_direction = 'declined'
    else:
        total_direction = 'unchanged'

    # 综合进步分 (0-100)
    avg_dim_change = sum(abs(r['change']) for r in radar_comparison) / 5
    overall_progress_score = min(100, max(0, 50 + avg_dim_change * 3))
    if total_direction == 'declined':
        overall_progress_score = max(0, overall_progress_score - 20)

    # --- 关键参数对比 ---
    param_config = [
        ('avg_torso_lean', '躯干前倾角', '°', 5, 10),
        ('knee_flex_min', '最小膝屈角', '°', 30, 50),
        ('knee_flex_max', '最大膝伸角', '°', 160, 180),
        ('avg_hip_angle', '平均髋角', '°', 150, 170),
        ('stride_ratio', '步幅比', '', 1.0, 2.5),
    ]
    params_comparison = []
    for key, name, unit, lo, hi in param_config:
        bv = before['params_summary'].get(key, 0)
        av = after['params_summary'].get(key, 0)
        change = round(av - bv, 1)
        # closer to ideal range middle = better
        ideal_mid = (lo + hi) / 2
        b_dist = abs(bv - ideal_mid)
        a_dist = abs(av - ideal_mid)
        diff_change = a_dist - b_dist  # negative means improved
        if diff_change < -0.5:
            direction = 'improved'
        elif diff_change > 0.5:
            direction = 'declined'
        else:
            direction = 'unchanged'
        params_comparison.append({
            'param_key': key,
            'param_name': name,
            'unit': unit,
            'before_value': bv,
            'after_value': av,
            'change': change,
            'direction': direction,
            'ideal_range': f'{lo}–{hi}{unit}'
        })

    # --- 问题对比 ---
    before_issue_labels = set()
    for iss in before.get('issues', []):
        label = iss.get('specific_error') or iss.get('dimension') or ''
        if label:
            before_issue_labels.add(label)
    # also add specific_errors
    for err in before.get('specific_errors', []):
        before_issue_labels.add(err.get('label', ''))

    after_issue_labels = set()
    for iss in after.get('issues', []):
        label = iss.get('specific_error') or iss.get('dimension') or ''
        if label:
            after_issue_labels.add(label)
    for err in after.get('specific_errors', []):
        after_issue_labels.add(err.get('label', ''))

    resolved = list(before_issue_labels - after_issue_labels)
    remaining = list(before_issue_labels & after_issue_labels)
    new_issues = list(after_issue_labels - before_issue_labels)

    # --- 步态相位对比 ---
    phase_config = [
        ('contact', '着地期'),
        ('stance', '支撑期'),
        ('push_off', '蹬伸期'),
        ('swing', '摆动期'),
    ]
    phase_comparison = {}
    for key, name in phase_config:
        bs = before['phase_scores'].get(key, 0)
        aft = after['phase_scores'].get(key, 0)
        phase_comparison[key] = {
            'phase_name': name,
            'before_score': round(bs, 1),
            'after_score': round(aft, 1),
            'change': round(aft - bs, 1)
        }

    # --- 自然语言评估 ---
    improved_dims = [r for r in radar_comparison if r['direction'] == 'improved']
    declined_dims = [r for r in radar_comparison if r['direction'] == 'declined']

    if total_direction == 'improved':
        summary = f'总体进步 {total_change:+.1f} 分！{"、".join(d["dimension_name"] for d in improved_dims[:3])} 方面有明显提升。'
    elif total_direction == 'declined':
        summary = f'总体下降 {total_change:+.1f} 分，需重点关注 {"、".join(d["dimension_name"] for d in declined_dims[:3])}。'
    else:
        summary = f'总体变化不大（{total_change:+.1f} 分），跑姿基本稳定。'

    strongest = improved_dims[0]['dimension_name'] if improved_dims else '无明显单项突出提升'
    needs_attn = declined_dims[0]['dimension_name'] if declined_dims else '各维度均保持良好'

    return {
        'overall_progress_score': round(overall_progress_score, 1),
        'total_score_change': total_change,
        'total_score_direction': total_direction,
        'radar_comparison': radar_comparison,
        'params_comparison': params_comparison,
        'issues_comparison': {
            'resolved_issues': resolved,
            'remaining_issues': remaining,
            'new_issues': new_issues
        },
        'phase_comparison': phase_comparison,
        'assessment': {
            'summary': summary,
            'strongest_improvement': strongest,
            'needs_attention': needs_attn
        }
    }


@app.route('/api/analyze_single_video', methods=['POST'])
def analyze_single_video():
    """分析单个视频并与标准库对比（无需上传标准视频）"""
    global detector, processing_lock

    if processing_lock:
        return jsonify({'status': 'error', 'message': '系统正在处理，请稍候'})

    if not detector or not detector.is_loaded():
        return jsonify({'status': 'error', 'message': '模型未加载'})

    processing_lock = True
    try:
        data = request.json
        video_path = data.get('video_path')
        if not video_path:
            return jsonify({'status': 'error', 'message': '请提供视频路径'})

        result = _run_single_video_analysis(
            detector, video_path,
            data.get('running_type'),
            data.get('gender')
        )

        if not result:
            return jsonify({'status': 'error', 'message': '视频分析失败，请确保视频中有清晰的人体'})

        return jsonify({'status': 'success', **result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'})
    finally:
        processing_lock = False


@app.route('/api/analyze_progress', methods=['POST'])
def analyze_progress():
    """进步检测：对比训练前后的两个视频"""
    global detector, processing_lock

    if processing_lock:
        return jsonify({'status': 'error', 'message': '系统正在处理，请稍候'})

    if not detector or not detector.is_loaded():
        return jsonify({'status': 'error', 'message': '模型未加载'})

    processing_lock = True
    try:
        data = request.json
        before_video = data.get('before_video')
        after_video = data.get('after_video')
        running_type = data.get('running_type')
        gender = data.get('gender')

        if not before_video or not after_video:
            return jsonify({'status': 'error', 'message': '请上传训练前和训练后的视频'})

        before_result = _run_single_video_analysis(detector, before_video, running_type, gender)
        if not before_result:
            return jsonify({'status': 'error', 'message': '训练前视频分析失败，请确保视频中有清晰的人体'})

        after_result = _run_single_video_analysis(detector, after_video, running_type, gender)
        if not after_result:
            return jsonify({'status': 'error', 'message': '训练后视频分析失败，请确保视频中有清晰的人体'})

        progress_data = _generate_progress_report(before_result, after_result)

        return jsonify({
            'status': 'success',
            'progress_data': progress_data,
            'before': before_result,
            'after': after_result
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'进步分析失败: {str(e)}'})
    finally:
        processing_lock = False

@app.route('/api/processed_video/<filename>')
def serve_processed_video(filename):
    """提供处理后视频的静态文件访问"""
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename),
                     mimetype='video/mp4')


@app.route('/sw.js')
def service_worker():
    """PWA Service Worker — 从根路径提供以获取全局作用域"""
    sw_path = os.path.join(app.static_folder, 'sw.js')
    resp = send_file(sw_path, mimetype='application/javascript')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/manifest.json')
def manifest_json():
    """PWA Manifest"""
    manifest_path = os.path.join(app.static_folder, 'manifest.json')
    return send_file(manifest_path, mimetype='application/manifest+json')


# Production entry: load model at import time (for gunicorn)
print("正在启动应用...")
load_model()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
