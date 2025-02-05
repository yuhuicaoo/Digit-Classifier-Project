import React from 'react'

function Header() {
  return (
    <section className="landing">
        <header>
            <div className="container">
                <div className="row">
                    <div className="content-wrapper">
                        <h1 className='header-title purple'>Digit Classifier</h1>
                        <h3 className='header-description purple'>Draw a digit for the model to predict</h3>
                    </div>
                </div>
            </div>
        </header>
    </section>
  )
}

export default Header