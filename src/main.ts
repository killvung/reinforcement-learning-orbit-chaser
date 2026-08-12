import Phaser from 'phaser';
import './styles.css';
import { PlayScene } from './game/PlayScene';

new Phaser.Game({
  type: Phaser.AUTO,
  parent: 'game',
  width: 800,
  height: 640,
  backgroundColor: '#071426',
  scene: [PlayScene],
  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: 800,
    height: 640,
  },
});
