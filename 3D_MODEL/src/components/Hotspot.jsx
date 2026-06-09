import React, { useCallback, useEffect, useState } from 'react';
import { useThree } from '@react-three/fiber';
import { Billboard, Text } from '@react-three/drei';
import { useSpring } from '@react-spring/three';
import * as THREE from 'three';
import '/src/App.css';

function Hotspot({
  position,
  text,
  dataKey = text,
  photoUrls = [],
  onCameraFocus,
  onHotspotSelect,
  telemetryFrames = 0
}) {
  const [isInfoVisible, setIsInfoVisible] = useState(false);
  const { camera } = useThree();
  const [targetPosition, setTargetPosition] = useState(null);
  const [resistance, setResistance] = useState([]);
  const [corrosion, setCorrosion] = useState([]);
  const [description, setDescription] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  const descriptions = {
    'Носовое подруливающее устройство':
      'Зона подвержена коррозии из-за постоянного контакта с водой и электрохимических процессов.',
    'Кормовой участок корпуса':
      'Участок кормы в зоне движительного комплекса. Здесь часто фиксируются обрастание, износ покрытия и локальные очаги коррозии.',
    'Гребной винт':
      'Эта область уязвима к гальванической коррозии и кавитационным повреждениям.',
    Киль: 'Коррозия развивается из-за длительного контакта с морской водой и механических нагрузок.'
  };

  const handleClick = () => {
    setTargetPosition(new THREE.Vector3(...position));
    setIsInfoVisible((previous) => !previous);
    if (onCameraFocus) {
      onCameraFocus(position);
    }
    if (onHotspotSelect) {
      onHotspotSelect({
        text,
        dataKey,
        photoUrls,
        telemetryFrames
      });
    }
    setDescription(descriptions[text] || '');
  };

  const fetchCSVData = useCallback(() => {
    return fetch('http://localhost:5001/api/data').then((response) => {
      if (!response.ok) {
        throw new Error('Ошибка запроса данных');
      }
      return response.json();
    });
  }, []);

  useEffect(() => {
    if (!isInfoVisible) {
      return;
    }

    fetchCSVData()
      .then((data) => {
        const hotspotData = data.find((item) => item.text === dataKey);
        if (!hotspotData) {
          setResistance(['Н/Д']);
          setCorrosion(['Н/Д']);
          return;
        }

        setResistance(
          Array.from({ length: 10 }, (_, index) => hotspotData[`resistance${index + 1}`] || 'Н/Д')
        );
        setCorrosion(
          Array.from({ length: 10 }, (_, index) => hotspotData[`corrosion${index + 1}`] || 'Н/Д')
        );
      })
      .catch((error) => {
        console.error('Ошибка чтения CSV:', error);
        setResistance(['Ошибка']);
        setCorrosion(['Ошибка']);
      });
  }, [dataKey, fetchCSVData, isInfoVisible]);

  useEffect(() => {
    if (!isInfoVisible) {
      return undefined;
    }

    const maxLength = Math.max(resistance.length, 1);
    const intervalId = window.setInterval(() => {
      setCurrentIndex((previousIndex) => (previousIndex + 1) % maxLength);
    }, 4000);

    return () => window.clearInterval(intervalId);
  }, [isInfoVisible, photoUrls.length, resistance.length]);

  const [, api] = useSpring(() => ({
    cameraPosition: camera.position,
    lookAtPosition: camera.position,
    config: { mass: 1, tension: 170, friction: 26 },
    onChange: ({ value }) => {
      camera.position.copy(value.cameraPosition);
      camera.lookAt(value.lookAtPosition);
    },
    onRest: () => setTargetPosition(null)
  }));

  useEffect(() => {
    if (!targetPosition) {
      return;
    }

    api.start({
      cameraPosition: targetPosition.clone().add(new THREE.Vector3(-5, 3, 6)),
      lookAtPosition: targetPosition
    });
  }, [api, targetPosition]);

  return (
    <group position={position} onClick={handleClick}>
      <mesh>
        <sphereGeometry args={[0.2, 32, 32]} />
        <meshStandardMaterial color="red" />
      </mesh>
      {isInfoVisible && (
        <Billboard>
          <group position={[-1.65, -1.25, 0]}>
            <mesh position={[1.65, 0, -0.01]}>
              <planeGeometry args={[3.5, 4.2]} />
              <meshBasicMaterial color="black" transparent opacity={0.68} />
            </mesh>
            <Text
              color="white"
              fontSize={0.28}
              fontWeight="bold"
              anchorX="left"
              anchorY="top"
              position={[0.12, 1.85, 0]}
              maxWidth={3.1}
              lineHeight={1.2}
            >
              {text}
            </Text>
            <Text
              color="white"
              fontSize={0.2}
              anchorX="left"
              anchorY="top"
              position={[0.12, 1.45, 0]}
              maxWidth={3.1}
              lineHeight={1.2}
            >
              {description}
            </Text>
            <Text
              color="#9ad8ff"
              fontSize={0.2}
              anchorX="left"
              anchorY="top"
              position={[0.12, 0.62, 0]}
              maxWidth={3.1}
            >
              Привязано снимков: {photoUrls.length}
            </Text>
            <Text
              color="#9ad8ff"
              fontSize={0.2}
              anchorX="left"
              anchorY="top"
              position={[0.12, 0.34, 0]}
              maxWidth={3.1}
            >
              Кадров телеметрии: {telemetryFrames}
            </Text>
            {resistance.length > 0 && (
              <Text
                color="yellow"
                fontSize={0.21}
                anchorX="left"
                anchorY="top"
                position={[0.12, -0.02, 0]}
                maxWidth={3.1}
              >
                Сопротивление: {resistance[currentIndex % resistance.length]} кОм
              </Text>
            )}
            {corrosion.length > 0 && (
              <Text
                color="yellow"
                fontSize={0.21}
                anchorX="left"
                anchorY="top"
                position={[0.12, -0.28, 0]}
                maxWidth={3.1}
              >
                Коррозия: {corrosion[currentIndex % corrosion.length]}
              </Text>
            )}
          </group>
        </Billboard>
      )}
    </group>
  );
}

export default Hotspot;
