import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { TextureLoader } from 'three';
import * as THREE from 'three';
import './App.css';

const INITIAL_CAMERA_POSITION = [-24, 10, 34];
const INITIAL_CAMERA_TARGET = [0, -0.8, 1.5];

const hotspotDefinitions = [
  { id: 'keel', text: '\u041a\u0438\u043b\u044c', dataKey: 'Keel', position: [-4.8, -1, 0] },
  { id: 'propeller', text: '\u0413\u0440\u0435\u0431\u043d\u043e\u0439 \u0432\u0438\u043d\u0442', dataKey: 'Propeller', position: [-4, -2, 17] },
  {
    id: 'bow-thruster',
    text: '\u041d\u043e\u0441\u043e\u0432\u043e\u0435 \u043f\u043e\u0434\u0440\u0443\u043b\u0438\u0432\u0430\u044e\u0449\u0435\u0435 \u0443\u0441\u0442\u0440\u043e\u0439\u0441\u0442\u0432\u043e',
    dataKey: 'Bow Thruster',
    position: [-1.5, -1, -14.5]
  },
  { id: 'engine', text: '\u041a\u043e\u0440\u043c\u043e\u0432\u043e\u0439 \u0443\u0447\u0430\u0441\u0442\u043e\u043a \u043a\u043e\u0440\u043f\u0443\u0441\u0430', dataKey: 'Engine', position: [-4.5, -1, 12] }
];

const hotspotDescriptionMap = {
  '\u041d\u043e\u0441\u043e\u0432\u043e\u0435 \u043f\u043e\u0434\u0440\u0443\u043b\u0438\u0432\u0430\u044e\u0449\u0435\u0435 \u0443\u0441\u0442\u0440\u043e\u0439\u0441\u0442\u0432\u043e':
    '\u0417\u043e\u043d\u0430 \u043f\u043e\u0434\u0432\u0435\u0440\u0436\u0435\u043d\u0430 \u043a\u043e\u0440\u0440\u043e\u0437\u0438\u0438 \u0438\u0437-\u0437\u0430 \u043f\u043e\u0441\u0442\u043e\u044f\u043d\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0430\u043a\u0442\u0430 \u0441 \u0432\u043e\u0434\u043e\u0439 \u0438 \u044d\u043b\u0435\u043a\u0442\u0440\u043e\u0445\u0438\u043c\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u043f\u0440\u043e\u0446\u0435\u0441\u0441\u043e\u0432.',
  '\u041a\u043e\u0440\u043c\u043e\u0432\u043e\u0439 \u0443\u0447\u0430\u0441\u0442\u043e\u043a \u043a\u043e\u0440\u043f\u0443\u0441\u0430':
    '\u0423\u0447\u0430\u0441\u0442\u043e\u043a \u043a\u043e\u0440\u043c\u044b \u0432 \u0437\u043e\u043d\u0435 \u0434\u0432\u0438\u0436\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0433\u043e \u043a\u043e\u043c\u043f\u043b\u0435\u043a\u0441\u0430. \u0417\u0434\u0435\u0441\u044c \u0447\u0430\u0441\u0442\u043e \u0444\u0438\u043a\u0441\u0438\u0440\u0443\u044e\u0442\u0441\u044f \u043e\u0431\u0440\u0430\u0441\u0442\u0430\u043d\u0438\u0435, \u0438\u0437\u043d\u043e\u0441 \u043f\u043e\u043a\u0440\u044b\u0442\u0438\u044f \u0438 \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u044b\u0435 \u043e\u0447\u0430\u0433\u0438 \u043a\u043e\u0440\u0440\u043e\u0437\u0438\u0438.',
  '\u0413\u0440\u0435\u0431\u043d\u043e\u0439 \u0432\u0438\u043d\u0442':
    '\u042d\u0442\u0430 \u043e\u0431\u043b\u0430\u0441\u0442\u044c \u0443\u044f\u0437\u0432\u0438\u043c\u0430 \u043a \u0433\u0430\u043b\u044c\u0432\u0430\u043d\u0438\u0447\u0435\u0441\u043a\u043e\u0439 \u043a\u043e\u0440\u0440\u043e\u0437\u0438\u0438 \u0438 \u043a\u0430\u0432\u0438\u0442\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u043c \u043f\u043e\u0432\u0440\u0435\u0436\u0434\u0435\u043d\u0438\u044f\u043c.',
  '\u041a\u0438\u043b\u044c': '\u041a\u043e\u0440\u0440\u043e\u0437\u0438\u044f \u0440\u0430\u0437\u0432\u0438\u0432\u0430\u0435\u0442\u0441\u044f \u0438\u0437-\u0437\u0430 \u0434\u043b\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0433\u043e \u043a\u043e\u043d\u0442\u0430\u043a\u0442\u0430 \u0441 \u043c\u043e\u0440\u0441\u043a\u043e\u0439 \u0432\u043e\u0434\u043e\u0439 \u0438 \u043c\u0435\u0445\u0430\u043d\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u043d\u0430\u0433\u0440\u0443\u0437\u043e\u043a.'
};

const hotspotSpecificPhotoMatchers = {
  'bow-thruster': (filename) => /^ruli(?: \(\d+\))?\.jpg$/i.test(filename),
  propeller: (filename) => /^propeller(?: \(\d+\))?\.jpg$/i.test(filename),
  engine: (filename) => /^stern_hull_\d+\.jpg$/i.test(filename)
};

const hotspotSpecificPhotoFiles = {
  'bow-thruster': ['ruli (1).jpg', 'ruli (2).jpg', 'ruli (3).jpg', 'ruli (4).jpg', 'ruli (5).jpg', 'ruli (6).jpg'],
  propeller: [
    'propeller (1).jpg',
    'propeller (2).jpg',
    'propeller (3).jpg',
    'propeller (4).jpg',
    'propeller (5).jpg',
    'propeller (6).jpg',
    'propeller (7).jpg',
    'propeller (8).jpg',
    'propeller (9).jpg',
    'propeller (10).jpg',
    'propeller (11).jpg',
    'propeller (12).jpg',
    'propeller (13).jpg'
  ],
  engine: [
    'stern_hull_01.jpg',
    'stern_hull_02.jpg',
    'stern_hull_03.jpg',
    'stern_hull_04.jpg',
    'stern_hull_05.jpg',
    'stern_hull_06.jpg',
    'stern_hull_07.jpg'
  ]
};

function Model({ useStitchedTexture, onModelLoaded }) {
  const gltf = useLoader(GLTFLoader, '/shipp.glb');
  const texture = useLoader(TextureLoader, '/generated/hull-texture/hull_texture.jpg');
  const clonedScene = useMemo(() => gltf.scene.clone(true), [gltf.scene]);

  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 8;
  }, [texture]);

  useEffect(() => {
    clonedScene.traverse((child) => {
      if (!child.isMesh) return;
      child.castShadow = true;
      child.receiveShadow = true;
      if (child.name === 'Hull_3') {
        child.material = child.material.clone();
        if (useStitchedTexture) {
          child.material.map = texture;
          child.material.needsUpdate = true;
        } else {
          child.material.map = null;
          child.material.needsUpdate = true;
        }
      }
    });
    onModelLoaded?.(clonedScene);
  }, [clonedScene, onModelLoaded, texture, useStitchedTexture]);

  return <primitive object={clonedScene} scale={1} />;
}

function CameraRig({ focusTarget, resetSignal, releaseSignal }) {
  const controlsRef = useRef(null);
  const [isResetting, setIsResetting] = useState(false);
  const [isReleasing, setIsReleasing] = useState(false);
  const targetVector = useMemo(() => (focusTarget ? new THREE.Vector3(...focusTarget) : null), [focusTarget]);

  useEffect(() => {
    setIsResetting(true);
  }, [resetSignal]);

  useEffect(() => {
    if (targetVector) {
      setIsResetting(false);
      setIsReleasing(false);
    }
  }, [targetVector]);

  useEffect(() => {
    if (!targetVector) {
      setIsReleasing(true);
    }
  }, [releaseSignal, targetVector]);

  useFrame(({ camera }) => {
    if (!controlsRef.current) {
      return;
    }

    if (isResetting) {
      const initialPosition = new THREE.Vector3(...INITIAL_CAMERA_POSITION);
      const initialTarget = new THREE.Vector3(...INITIAL_CAMERA_TARGET);
      camera.position.lerp(initialPosition, 0.08);
      controlsRef.current.target.lerp(initialTarget, 0.08);
      controlsRef.current.update();

      if (camera.position.distanceTo(initialPosition) < 0.04 && controlsRef.current.target.distanceTo(initialTarget) < 0.04) {
        camera.position.copy(initialPosition);
        controlsRef.current.target.copy(initialTarget);
        controlsRef.current.update();
        setIsResetting(false);
      }
      return;
    }

    if (isReleasing) {
      const initialTarget = new THREE.Vector3(...INITIAL_CAMERA_TARGET);
      controlsRef.current.target.lerp(initialTarget, 0.12);
      controlsRef.current.update();

      if (controlsRef.current.target.distanceTo(initialTarget) < 0.04) {
        controlsRef.current.target.copy(initialTarget);
        controlsRef.current.update();
        setIsReleasing(false);
      }
      return;
    }

    if (!targetVector) {
      return;
    }

    const desiredPosition = targetVector.clone().add(new THREE.Vector3(-5, 3, 6));
    camera.position.lerp(desiredPosition, 0.06);
    controlsRef.current.target.lerp(targetVector, 0.08);
    controlsRef.current.update();
  });

  return <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.08} />;
}

function Hotspots({ hotspotPhotoMap, onHotspotSelect, onFocus }) {
  return hotspotDefinitions.map((hotspot) => (
    <mesh
      key={hotspot.id}
      position={hotspot.position}
      onClick={(event) => {
        event.stopPropagation();
        onFocus(hotspot.position);
        onHotspotSelect({
          ...hotspot,
          photoUrls: hotspotPhotoMap[hotspot.id] || []
        });
      }}
    >
      <sphereGeometry args={[0.18, 24, 24]} />
      <meshStandardMaterial color="#d72638" emissive="#7f0814" emissiveIntensity={0.45} />
    </mesh>
  ));
}

export default function App() {
  const [selectedHotspot, setSelectedHotspot] = useState(null);
  const [focusTarget, setFocusTarget] = useState(null);
  const [resetSignal, setResetSignal] = useState(0);
  const [releaseSignal, setReleaseSignal] = useState(0);
  const [textureMeta, setTextureMeta] = useState(null);
  const [useStitchedTexture, setUseStitchedTexture] = useState(true);

  const closeHotspot = () => {
    setSelectedHotspot(null);
    setFocusTarget(null);
    setReleaseSignal((value) => value + 1);
  };

  const resetCamera = () => {
    setSelectedHotspot(null);
    setFocusTarget(null);
    setResetSignal((value) => value + 1);
  };

  useEffect(() => {
    fetch(`/generated/hull-texture/hull_texture_manifest.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => setTextureMeta(payload))
      .catch(() => setTextureMeta(null));
  }, []);

  const hotspotPhotoMap = useMemo(() => {
    const assignments = {};
    hotspotDefinitions.forEach((hotspot) => {
      assignments[hotspot.id] = [];
    });

    const sources = textureMeta?.sources || [];
    if (sources.length === 0) {
      return assignments;
    }

    const reservedFiles = new Set();
    Object.entries(hotspotSpecificPhotoMatchers).forEach(([hotspotId, matcher]) => {
      const fallbackFiles = hotspotSpecificPhotoFiles[hotspotId] || [];
      const matched = sources.filter((filename) => matcher(filename));
      const finalFiles = matched.length > 0 ? matched : fallbackFiles;
      assignments[hotspotId] = finalFiles.map((filename) => `/generated/hotspot-photos/${filename}`);
      finalFiles.forEach((filename) => reservedFiles.add(filename));
    });

    const remainingHotspots = hotspotDefinitions.filter((hotspot) => !hotspotSpecificPhotoMatchers[hotspot.id]);
    const remainingSources = sources.filter((filename) => !reservedFiles.has(filename));
    const chunkSize = remainingHotspots.length > 0 ? Math.ceil(remainingSources.length / remainingHotspots.length) : 0;

    remainingHotspots.forEach((hotspot, hotspotIndex) => {
      const start = hotspotIndex * chunkSize;
      const end = start + chunkSize;
      assignments[hotspot.id] = remainingSources.slice(start, end).map((filename) => `/generated/hotspot-photos/${filename}`);
    });

    return assignments;
  }, [textureMeta]);

  return (
    <div className="app-shell">
      <div className="model-toolbar">
        <button className="toolbar-button" onClick={() => setUseStitchedTexture((previous) => !previous)}>
          {useStitchedTexture ? '\u041f\u043e\u043a\u0430\u0437\u0430\u0442\u044c \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u043a\u043e\u0440\u043f\u0443\u0441' : '\u041f\u043e\u043a\u0430\u0437\u0430\u0442\u044c \u0441\u043e\u0431\u0440\u0430\u043d\u043d\u0443\u044e \u0442\u0435\u043a\u0441\u0442\u0443\u0440\u0443'}
        </button>
        <button className="toolbar-button toolbar-button--secondary" onClick={resetCamera}>
          Вернуть начальный вид
        </button>
        <div className="toolbar-meta">
          <strong>Hull_3</strong>
          <span>Кадров с признаками: {textureMeta?.sourceCount || 0}</span>
        </div>
      </div>

      <div className="canvas-stage">
        <Canvas camera={{ position: INITIAL_CAMERA_POSITION, fov: 48 }} shadows>
          <color attach="background" args={['#eef4fa']} />
          <ambientLight intensity={1.6} />
          <directionalLight position={[8, 10, 6]} intensity={2.4} castShadow />
          <Suspense fallback={null}>
            <Model useStitchedTexture={useStitchedTexture} />
            <Hotspots hotspotPhotoMap={hotspotPhotoMap} onHotspotSelect={setSelectedHotspot} onFocus={setFocusTarget} />
          </Suspense>
          <gridHelper args={[50, 40, '#c7d3df', '#dbe4ee']} position={[0, -3.3, 0]} />
          <CameraRig focusTarget={focusTarget} resetSignal={resetSignal} releaseSignal={releaseSignal} />
        </Canvas>
      </div>

      {selectedHotspot && (
        <aside className="photo-sidebar open">
          <div className="photo-sidebar__header">
            <div>
              <h3>{selectedHotspot.text}</h3>
              <p>{hotspotDescriptionMap[selectedHotspot.text] || '\u0421\u0432\u044f\u0437\u0430\u043d\u043d\u044b\u0435 \u0441\u043d\u0438\u043c\u043a\u0438 hotspot.'}</p>
              <p>Привязано снимков: {selectedHotspot.photoUrls?.length || 0}</p>
            </div>
            <button className="photo-sidebar__close" onClick={closeHotspot}>
              Закрыть
            </button>
          </div>
          <div className="photo-sidebar__list">
            {(selectedHotspot.photoUrls || []).map((photoUrl, index) => (
              <figure key={`${selectedHotspot.id}-${index}`} className="photo-sidebar__item">
                <img src={`${photoUrl}?t=${Date.now()}`} alt={`${selectedHotspot.text} ${index + 1}`} />
                <figcaption>Кадр {index + 1}</figcaption>
              </figure>
            ))}
          </div>
        </aside>
      )}
    </div>
  );
}
