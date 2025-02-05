import { useState } from 'react';
import './App.css';
import Canvas from './components/Canvas';
import Header from './components/Header';

function App() {
  const [prediction, setPrediction] = useState(null);

  return (
    <div className="App">
      <Header/>
      <Canvas setPrediction={setPrediction}/>
    </div>
  );
}

export default App;
