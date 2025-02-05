import React from "react";

function Prediction({ prediction, isLoading }) {
  const { digit = "-", probabilities = new Array(10).fill("-") } =
    prediction ?? {};

  return (
    <div className="container">
      <div className="row">
        <div className="content-wrapper">
          <div className="prediction">
            <h2 className="purple">Prediction: {digit}</h2>
          </div>
          <div className="confidence">
            <h3 className="confidence-header purple">Confidence</h3>
            <div className="confidence-table">
              {probabilities.map((prob, index) => (
                <div key={index}>
                  <div className="digit">{index}</div>
                  <div className="prob">{prob} %</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Prediction;
