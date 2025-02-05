import React, { useEffect, useRef, useCallback } from "react";
import axios from "axios";

function Canvas({ setPrediction }) {
    const canvasRef = useRef(null);
    const isDrawing = useRef(false);

    const getCoords = (event) => {
        const { pageX, pageY } = event.changedTouches ? event.changedTouches[0] : event;
        const canvas = canvasRef.current;
        return { x: pageX - canvas.offsetLeft, y: pageY - canvas.offsetTop };
    };


    const draw = useCallback((event, start) => {
        event.preventDefault();
        const context = canvasRef.current?.getContext("2d");
        if (!context) return;

        const { x, y } = getCoords(event);
        if (start) {
            isDrawing.current = true;
            context.beginPath();
            context.moveTo(x, y);
        } else if (isDrawing.current) {
            context.lineTo(x, y);
            context.strokeStyle = "white";
            context.lineWidth = 10;
            context.lineCap = "round";
            context.lineJoin = "round";
            context.stroke();
        }
    }, []);

    const stopDrawing = useCallback((event) => {
        event.preventDefault();
        if (isDrawing.current) {
            const context = canvasRef.current?.getContext("2d");
            context?.closePath();
            isDrawing.current = false;
        }
    }, []);

    const clearCanvas = () => {
        const canvas = canvasRef.current;
        const context = canvas?.getContext("2d");
        if (canvas && context) {
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.fillStyle = "black";
            context.fillRect(0, 0, canvas.width, canvas.height);
        }
    };

    const predict = async () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const imageDataURL = canvas.toDataURL("image/png");

        try {
            const response = await axios.post("http://localhost:5000/predict", {
                image: imageDataURL,
            });

            setPrediction(response.data.prediction);
        } catch (error) {
            console.error("Prediction failed:", error.response?.data || error.message);
        }
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const events = [
            { name: "mousedown", handler: (e) => draw(e, true) },
            { name: "touchstart", handler: (e) => draw(e, true) },
            { name: "mousemove", handler: (e) => draw(e, false) },
            { name: "touchmove", handler: (e) => draw(e, false) },
            { name: "mouseup", handler: stopDrawing },
            { name: "touchend", handler: stopDrawing },
        ];

        events.forEach(({ name, handler }) => canvas.addEventListener(name, handler, { passive: false }));
        return () => events.forEach(({ name, handler }) => canvas.removeEventListener(name, handler));
    }, [draw, stopDrawing]);
    useEffect(() => {
        const handleGlobalEnd = (event) => isDrawing.current && stopDrawing(event);
        window.addEventListener("mouseup", handleGlobalEnd, { passive: false });
        window.addEventListener("touchend", handleGlobalEnd, { passive: false });
        return () => {
            window.removeEventListener("mouseup", handleGlobalEnd);
            window.removeEventListener("touchend", handleGlobalEnd);
        };
    }, [stopDrawing]);

    return (
        <section className="prediction">
            <div className="container">
                <div className="row">
                    <div className="content-wrapper">
                        <canvas ref={canvasRef} width={280} height={280} className="canvas" />
                        <div className="button-wrapper">
                            <button className="btn" onClick={clearCanvas}>Clear</button>
                            <button className="btn" onClick={predict}>Predict</button>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}

export default Canvas;
