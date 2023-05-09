import './App.css';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Card from 'react-bootstrap/Card';
//import SentimentBar from './Bar';
import Stack from 'react-bootstrap/Stack';
import React from 'react';
import ProgressBar from 'react-bootstrap/ProgressBar';

function App() {
  const [text, updateText] = React.useState("");
  const dummyQuery = () => Math.round(100*Math.random())

  const sentiments = [
    { "labels": ["negative", "positive"], "color": "success", "val": React.useState(50), "queryFn": dummyQuery, "barRef": null },
    { "labels": ["sarcastic", "sincere"], "color": "info", "val": React.useState(50), "queryFn": dummyQuery, "barRef": null  },
    { "labels": ["ambiguous", "concrete"], "color": "warning", "val": React.useState(50), "queryFn": dummyQuery, "barRef": null  },
    {"labels": ["calm", "intense"], "color": "danger", "val": React.useState(50), "queryFn": dummyQuery, "barRef": null }
  ]

  function TextInput() {
    const onFormSubmit = e => {
      e.preventDefault()
      const formData = new FormData(e.target),
            formDataObj = Object.fromEntries(formData.entries())
      console.log(formDataObj)
      handleSubmit(formDataObj.text)
    }

    return (
    <Form onSubmit={onFormSubmit}>
      <Form.Group className="mb-3">
        <Form.Control placeholder="Enter text here" defaultValue={text} style={{ minHeight: '20vh'}} as="textarea" rows="3" name="text"/>
        </Form.Group>
      <Stack direction="horizontal" gap={3}>
        <Button variant="primary" type="submit" size="lg">
          Submit
          </Button>
        <Button variant="danger" onClick={()=>reset()} size="lg">
          Clear
        </Button>
      </Stack>
    </Form>
    )
  }

  function reset() {
    updateText("")
    sentiments.forEach((s, i, arr) => sentiments[i].val[1](0))
  }

  function SentimentGroup(sentiment , i) {
    const elementRef = React.useRef();
    sentiments[i]['barRef'] = elementRef
    return (
      <Stack direction="horizontal" gap={3}>
        <p style={{align: "left", width: "100px"}}>{sentiment.labels[0].toUpperCase()}</p>
        <ProgressBar ref={elementRef} now={sentiment.val[0]} variant={sentiment.color} style={{width: '350px'}}/>
        <p style={{align: "right", width: "auto"}}>{sentiment.labels[1].toUpperCase()}</p>
      </Stack>
    )
  }

  function handleSubmit(input) {
    //console.log(input)
    updateText(input)
    sentiments.forEach((s, i, arr) => sentiments[i].val[1](s.queryFn(input)))
    sentiments.forEach((s, i, arr) => console.log(s.val[0]))
  }

  React.useEffect(() => {
    //textRef.current = text;
    //console.log(text)
  });

  return (
    <div className="App">
      <div className='App-header'>
        <h1>Decoji</h1>
        <Card style={{ width: '600px' }} border="primary">
          <Card.Body>
            <TextInput />
          </Card.Body>
          <Card.Footer>
              <Stack id="bars "gap={3}>
                {sentiments.map(SentimentGroup)}
              </Stack>
          </Card.Footer>
        </Card>
      </div>
    </div>
  );
}

export default App;
