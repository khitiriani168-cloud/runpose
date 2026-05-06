const CACHE_NAME = 'pose-app-v1';

// 需要缓存的关键静态资源
const PRECACHE_URLS = [
    '/static/manifest.json',
    '/static/icon-192.png',
    '/static/icon-512.png',
    '/static/loading.mp4',
    '/static/bg.png'
];

// 安装：预缓存静态资源
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(PRECACHE_URLS);
        })
    );
    self.skipWaiting();
});

// 激活：清理旧缓存
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(names => {
            return Promise.all(
                names.filter(name => name !== CACHE_NAME)
                    .map(name => caches.delete(name))
            );
        })
    );
    self.clients.claim();
});

// 拦截请求：网络优先，缓存作为后备
self.addEventListener('fetch', event => {
    // 只缓存 GET 请求
    if (event.request.method !== 'GET') return;

    // API 请求和 HTML 页面不缓存，直接走网络
    if (event.request.url.includes('/api/') || event.request.mode === 'navigate') {
        event.respondWith(fetch(event.request).catch(() => {
            return caches.match('/');
        }));
        return;
    }

    // 静态资源：缓存优先
    event.respondWith(
        caches.match(event.request).then(cached => {
            return cached || fetch(event.request).then(response => {
                // 缓存成功的响应
                if (response && response.status === 200) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            });
        })
    );
});
