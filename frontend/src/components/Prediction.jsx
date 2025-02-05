import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSpinner } from "@fortawesome/free-solid-svg-icons";

function Prediction({ prediction, isLoading }) {
  const { digit = "-", probabilities = new Array(10).fill("-") } =
    prediction ?? {};

  return (
    <div className="container">
      <div className="row">
        <div className="content-wrapper">
          <div className="prediction">
            <h2 className="purple">
              Prediction : {""}
              {isLoading ? (
                <FontAwesomeIcon
                  icon={faSpinner}
                  spin
                  className="ml-2"
                  size="xs"
                />
              ) : (
                <span>{digit}</span>
              )}
            </h2>
          </div>
          <div className="probabilities">
            <h3 className="probabilities__header purple">Probabilities</h3>
            {isLoading ? (
              <div>
                <div className="skeleton skeleton__table">
                  <div className="skeleton__table__spinner">
                    <FontAwesomeIcon
                      icon={faSpinner}
                      spin
                      className="ml-2"
                      size="2x"
                    />

                  </div>
                </div>
              </div>
            ) : (
              <div className="probabilities__table">
                {probabilities.map((prob, index) => (
                  <div className="value__container" key={index}>
                    <div className="digit">{index}</div>
                    <div className="prob">{prob}%</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Prediction;
