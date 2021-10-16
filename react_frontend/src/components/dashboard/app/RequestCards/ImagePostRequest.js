// material
import { React, useState } from 'react';
import PropTypes from 'prop-types';
import { styled } from '@material-ui/core/styles';
import {
  Divider,
  Card,
  Grid,
  CardContent,
  Alert,
  AlertTitle,
  Typography,
  Stack,
  Button,
  Box,
  ButtonGroup,
  Accordion,
  AccordionDetails,
  AccordionSummary,
  LinearProgress
} from '@material-ui/core';
// utils

import { sendPostRequest, getImageUrl } from '../../../../API/component';
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
const ImgStyle = styled('img')({
  top: 0,
  width: '30vw',
  margin: '4%',
  marginLeft: '1%',
  alignItems: 'center',
  borderRadius: '25px'
});

export default function UploadPicture(props) {
  const [imgSrc, setImgSrc] = useState('');
  const [uploadFile, setUploadFile] = useState({});
  const [content, setContent] = useState();
  const [loading, setLoading] = useState(false);

  const _onChange = (e) => {
    if (e.target.files.length !== 0) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImgSrc(url);
      setUploadFile(file);
    }
  };

  const handleFormUpload = () => {
    const formData = new FormData();
    formData.append('image', uploadFile);
    formData.append('filename', uploadFile.name);
    const config = {
      headers: {
        'content-type': 'multipart/form-data'
      }
    };
    sendPostRequest(
      props.api.endpoint,
      formData,
      (e) => {
        const key = Object.keys(e)[0];

        if (props.hideResponse !== true) {
          if (props.hideResponse !== true) {
            setContent(JSON.stringify(e, null, 2));
          }
        } else {
          setContent('Success');
        }
        getImageUrl(e[key].overlay_image, setImgSrc);
        if (props.callback) {
          props.callback(e);
        }
      },
      config
    );
  };

  const ProgressOrNothing = (progress) =>
    progress ? (
      <LinearProgress color="secondary" sx={{ m: '3%', height: '2vh', borderRadius: '5px' }} />
    ) : null;

  return (
    <Card
      variant="outlined"
      sx={{ m: '2%', borderRadius: '10px', boxShadow: 'none', textAlign: 'left' }}
    >
      <CardContent>
        <Stack direction="row">
          <Typography variant="h5" component="h2">
            {props.api.input}
          </Typography>
        </Stack>
        <Divider sx={{ my: '1%' }} />
        <Box>
          <Box>
            {content === undefined ? (
              ProgressOrNothing(loading)
            ) : (
              <Alert sx={{ m: '1.5%', overflow: 'hidden' }} severity="info">
                <pre>{content}</pre>
              </Alert>
            )}
          </Box>
          <Grid container justify>
            {imgSrc ? (
              <Stack>
                <Typography variant="subtitle1"> {uploadFile.name}</Typography>
                <ImgStyle src={imgSrc} alt="" />
              </Stack>
            ) : null}
          </Grid>

          <ButtonGroup>
            <Button variant="text" component="label" color="info" size="large">
              Upload Picture
              <input
                type="file"
                hidden
                onChange={(e) => {
                  _onChange(e);
                }}
              />
            </Button>
            <Button
              variant="text"
              component="label"
              color="info"
              size="large"
              disabled={!imgSrc || loading}
              onClick={() => {
                setContent(undefined);
                setLoading(true);
                handleFormUpload();
              }}
            >
              Submit
            </Button>
          </ButtonGroup>
          <Box flex="1" />
        </Box>
        <Accordion
          disableGutters
          sx={{
            '&:before': {
              display: 'none'
            },
            marginLeft: '-1%'
          }}
        >
          <AccordionSummary>
            <Button color="info" variant="outlined">
              Python Code Sample
            </Button>
          </AccordionSummary>
          <AccordionDetails>
            <Alert severity="secondary">
              <p>import requests</p>
              <br />
              <p>baseURL = %BASEURL%</p>
              <p>filename = %FILENAME%</p>
              <p>filepath = %FILEPATH% </p>

              <br />
              <p>
                files=[ ('image',(filename,open(filepath,'rb'),'image/jpeg')) ]
                <br />
                headers = {'{}'}
                <br />
                payload = {'{"filename": [filename]}'}
                <br />
                <br />
                response = requests.post( baseURL + "
                {props.api.endpoint_display.replace('<id>', '%ID%')}", headers=headers,
                data=payload, files=files)
                <br />
                print(response.json(), flush=True)
              </p>
            </Alert>
          </AccordionDetails>
        </Accordion>
      </CardContent>
    </Card>
  );
}
UploadPicture.propTypes = {
  api: PropTypes.object
};
