import { useState } from 'react';
import './index.css';
import Canvas from './components/Canvas';
import Header from './components/Header';
import Prediction from './components/Prediction';

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState(null);
  return (
    <div className="App">
      <Header/>
      <Canvas setPrediction={setPrediction} setIsLoading={setIsLoading}/>
      <Prediction prediction={prediction} isLoading={isLoading}/>
    </div>
  );
}

export default App;
